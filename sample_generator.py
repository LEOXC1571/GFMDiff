
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '10452'
import yaml
import time
import argparse
import wandb
start_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())

import warnings
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from models import model_map
from evaluator import analyze_stability_for_molecules, check_stability
from utils import init_seeds, BasicMolecularMetrics
import utils.visualization as vis

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OUTPUT_PATH = os.path.join(CURRENT_PATH, 'outputs')

warnings.filterwarnings('ignore')


def sample_various_mol(model, model_config, dataset_config, device, n_samples=100, batch_size=100, context=None):
    batch_size = min(n_samples, batch_size)
    model.eval()
    pos, onehot, atom_num, degree, node_mask = model.module.sample(batch_size, dataset_config['max_n_nodes'], device, context)

    vis.save_xyz_file(os.path.join(OUTPUT_PATH, 'various/{dataset}_{date}/'.
                                   format(dataset=dataset_config['name'], date=start_time)),
                      one_hot=onehot, charges=atom_num, positions=pos,
                      dataset_info=dataset_config, node_mask=node_mask)


def sample_stable_mol(model, model_config, dataset_config, device, context=None, num_attempt=2, calc_mol=False):
    n_samples = 1
    batch_size = num_attempt
    model.eval()
    pos, onehot, atom_num, degree, node_mask = model.module.sample(batch_size, dataset_config['max_n_nodes'], device, context)
    counter = 0
    smiles_list, mol_list = None, None
    oh_list, c_list, p_list, n_mask = [], [], [], []
    for i in range(num_attempt):
        num_atoms = int(node_mask[i:i+1].sum().item())
        atom_type = onehot[i:i+1, :num_atoms].argmax(2).squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(pos[i:i+1, :num_atoms].cpu().detach(), atom_type, dataset_config)[0]

        num_remaining_attempts = num_attempt - i - 1
        num_remaining_samples = n_samples - counter
        if calc_mol:
            metrics = BasicMolecularMetrics(dataset_config)
            pos_valid = pos[-1].cpu().detach()
            onehot_valid = onehot[-1].argmax(1).cpu().detach()
            valid, validity = metrics.compute_validity([(pos_valid, onehot_valid)])
            if validity == 1:
                smiles_list, mol_list = metrics.compute_mol([(pos_valid, onehot_valid)])
            else:
                smiles_list, mol_list = None, None
        if mol_stable or num_remaining_attempts <= num_remaining_samples:
            print('Found stable mol.')
            oh_list.append(onehot[i:i+1])
            c_list.append(atom_num[i:i+1])
            p_list.append(pos[i:i+1])
            n_mask.append(node_mask[i:i+1])
            counter += 1
            if counter >= n_samples:
                break
    onehot_all = torch.cat(oh_list, dim=0)
    c_all = torch.cat(c_list, dim=0)
    p_all = torch.cat(p_list, dim=0)
    mask_all = torch.cat(n_mask, dim=0)
    vis.save_xyz_file(
        os.path.join(OUTPUT_PATH, 'stable/{dataset}_{date}/'.
                        format(dataset=dataset_config['name'], date=start_time)),
        one_hot=onehot_all, charges=c_all, positions=p_all,
        dataset_info=dataset_config, node_mask=mask_all)
    vis.visualize(os.path.join(OUTPUT_PATH, 'stable/{dataset}_{date}/'.format(dataset=dataset_config['name'], date=start_time)),
                    dataset_config, max_num=100, spheres_3d=True)
    print('Done')


def sample_vis_chain(model, model_config, dataset_config, device, context=None,
                     num_chain=100, num_attempt=10, calc_mol=False):
    for i in range(num_chain):
        path = os.path.join(OUTPUT_PATH, 'chain/{dataset}_{date}/{chain}/'.
                            format(dataset=dataset_config['name'], date=start_time, chain=i))
        os.makedirs(path)
        n_samples = 1
        if dataset_config['name'] == 'qm9':
            n_nodes = 19
        elif dataset_config['name'] == 'drugs':
            n_nodes = 44
        else:
            raise ValueError('Unrecognized dataset: %s' % dataset_config['name'])

        smiles_list, mol_list = None, None
        for j in range(num_attempt):
            chain = model.module.sample_chain(n_samples, n_nodes, device, context, fix_noise=False, keep_frames=100)
            chain = chain[torch.arange(chain.size(0) - 1, -1, -1)]
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            pos_0 = chain[-1:, :, 0:3]
            onehot_0 = chain[-1:, :, 3:-2]
            onehot_0 = torch.argmax(onehot_0, dim=2)
            atom_type = onehot_0.squeeze(0).cpu().detach().numpy()
            pos_squeeze = pos_0.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(pos_squeeze, atom_type, dataset_config)[0]

            pos = chain[:, :, 0:3]
            onehot = chain[:, :, 3:-2]
            onehot = F.one_hot(torch.argmax(onehot, dim=2), num_classes=len(dataset_config['atom_decoder']))
            charges = torch.round(chain[:, :, -1:]).long()
            metrics = BasicMolecularMetrics(dataset_config)
            if calc_mol:
                pos_valid = pos[-1].cpu().detach()
                onehot_valid = onehot[-1].argmax(1).cpu().detach()
                valid, validity = metrics.compute_validity([(pos_valid, onehot_valid)])
                if validity == 1:
                    smiles_list, mol_list = metrics.compute_mol([(pos_valid, onehot_valid)])
                else:
                    smiles_list, mol_list = None, None
            if smiles_list is not None:
                with(open(os.path.join(OUTPUT_PATH, 'chain/{dataset}_{date}/'.format(dataset=dataset_config['name'], date=start_time), 'chain' + str(i) + '_smiles.txt'), "w")) as smiles_f:
                    try:
                        smiles_f.write(smiles_list[0])
                        Draw.MolToFile(mol_list[0],  path + 'chain_rdkit_withH.png', size=(500, 500))
                        Draw.MolToFile(Chem.RemoveHs(mol_list[0]),  path + 'chain_rdkit_noH.png', size=(500, 500))
                    except:
                        smiles_f.write(' ')
                    smiles_f.close()
            if mol_stable:
                print("Found stable molecule to visualize!")
                break
            elif j == num_attempt - 1:
                print("Did not find stable molecule, showing last sample...")

        vis.save_xyz_file(path, one_hot=onehot, charges=charges, positions=pos, dataset_info=dataset_config,
                          id_from=0, name='chain', smiles_list=smiles_list, mol_list=mol_list, i=i)
        vis.visualize_chain_uncertainty(path, dataset_config, spheres_3d=True)


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
    if args.task == 'random':
        print("Generating handful of molecules") if rank == 0 else None
        sample_various_mol(model, model_config, dataset_config, device)
    elif args.task == 'stable':
        print("Generating handful of stable molecules") if rank == 0 else None
        sample_stable_mol(model, model_config, dataset_config, device)
    elif args.task == 'chain':
        print("Visualizing molecules...") if rank == 0 else None
        sample_vis_chain(model, model_config, dataset_config, device, context=None, num_chain=10, num_attempt=10, calc_mol=True)
        vis.visualize(os.path.join(OUTPUT_PATH, 'stable'), dataset_config, max_num=100, spheres_3d=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='gfmdiff', action='store',
                        help="molecular graph generation models")
    parser.add_argument("--data", type=str, default="qm9", action='store',
                        help="the training data")
    parser.add_argument("--task", type=str, default="stable", action='store')
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--ckpt_dir", type=str, action='store')
    parser.add_argument('--wandb', action='store_true', default=False)
    args, unknown = parser.parse_known_args()
    os.environ['NCCL_SHM_DISABLE'] = '1'
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
