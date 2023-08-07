
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '10125'
import yaml
import time
import argparse
import wandb
start_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())

import warnings
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from models import model_map
from evaluator import analyze_stability_for_molecules
from utils import init_seeds

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OUTPUT_PATH = os.path.join(CURRENT_PATH, 'output/sample')

warnings.filterwarnings('ignore')


def analyze(model, model_config, dataset_config, device, n_samples=10000, batch_size=50, rank=0):
    model.eval()
    disable_tqdm = rank != 0
    batch_size = min(batch_size, n_samples)
    molecules = {'pos': [], 'onehot': [], 'node_mask': []}
    context = None
    tqdm_bar = tqdm(range(int(n_samples / batch_size)), desc="Iteration", disable=disable_tqdm)
    for i in tqdm_bar:
        with torch.no_grad():
            pos, onehot, atom_num, degree, node_mask = model.module.sample(batch_size, dataset_config['max_n_nodes'],
                                                                           device, context)
            molecules['pos'].append(pos.detach().cpu())
            molecules['onehot'].append(onehot.detach().cpu())
            molecules['node_mask'].append(node_mask.detach().cpu())
    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity, rdkit_metrics, rdkit_unique = analyze_stability_for_molecules(molecules, dataset_config)
    return validity, rdkit_metrics, rdkit_unique


def main(rank, world_size, args):
    model_name = args.model
    dataset_name = args.data
    use_wandb = args.wandb
    model_config = yaml.load(open(os.path.join(CURRENT_PATH, 'config/model/' + model_name + '.yaml'), "r"),
                             Loader=yaml.FullLoader)
    dataset_config = yaml.load(open(os.path.join(CURRENT_PATH, 'config/dataset/' + dataset_name + ".yaml"), "r"),
                               Loader=yaml.FullLoader)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    if torch.cuda.is_available():
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device('cpu')

    init_seeds(model_config['seed'])
    if rank == 0 and use_wandb:
        wandb.init(config={**model_config, **dataset_config},
                   project=start_time + '_' + args.comment,
                   name=model_name + '_on_' + dataset_name + '_sample')
        wandb.save('*.txt')

    model_class = model_map[model_name]
    model = model_class(model_config, dataset_config).to(device)

    ckpt = torch.load(args.ckpt_dir)
    model.load_state_dict(ckpt['model_state_dict'])
    model = DistributedDataParallel(model, device_ids=[rank])
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model successfully loaded, Number of Params: {num_params}')

    dist.barrier()
    print('Analyzing...') if rank == 0 else None
    validity, rdkit_metrics, rdkit_unique = analyze(model, model_config, dataset_config, device,
                                                    n_samples=10000/world_size, batch_size=50, rank=rank)
    validity, rdkit_metrics = validity.to(device), rdkit_metrics.to(device)
    validity_gather_list = [torch.zeros_like(validity) for _ in range(world_size)]
    rdkit_gather_list = [torch.zeros_like(rdkit_metrics) for _ in range(world_size)]
    dist.all_gather(validity_gather_list, validity)
    dist.all_gather(rdkit_gather_list, rdkit_metrics)
    validity = torch.cat(validity_gather_list, dim=0).mean(0) if rank == 0 else None
    rdkit_metrics = torch.cat(rdkit_gather_list, dim=0).mean(0) if rank == 0 else None
    analyze_dict = {
        'mol_stale': validity[0].item(),
        'atom_stable': validity[1].item(),
        'rdkit_validity': rdkit_metrics[0].item(),
        'rdkit_uniqueness': rdkit_metrics[1].item(),
        'rdkit_novelty': rdkit_metrics[2].item()
    } if rank == 0 else None
    if use_wandb and rank == 0:
        wandb.log(analyze_dict)
    torch.distributed.destroy_process_group()
    print(analyze_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='gfmdiff', action='store',
                        help="molecular graph generation models")
    parser.add_argument("--data", type=str, default="qm9", action='store',
                        help="the training data")
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--ckpt_dir", type=str, action='store')
    parser.add_argument('--wandb', action='store_true', default=False)
    args, unknown = parser.parse_known_args()
    os.environ['NCCL_SHM_DISABLE'] = '1'
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
