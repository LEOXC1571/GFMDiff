
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '18017'
import yaml
import time
import copy
import argparse
import wandb
start_time = time.strftime('%Y-%m-%d_%H-%M-%S_', time.localtime())

import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from models import model_map
from data import dataset_map
from evaluator import analyze_stability_for_molecules
from utils import init_seeds, EMA, Queue, gradient_clipping, DistributionProperty, write_log
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_DEBUG_SUBSYS"] = "COLL"
# os.environ["NCCL_DEBUG_FILE"] = "/output/nccl_logs.txt"
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
warnings.filterwarnings('ignore')


# def train(model, loader, optimizer, device, model_config, disable_tqdm, model_ema, ema, gradnorm_queue=None, use_wandb=False):
#     model.train()
#     nll_epoch, n_samples = [], 0
#     iter_size = 16
#     tqdm_bar = tqdm(loader, desc="Iteration", disable=disable_tqdm)
#     optimizer.zero_grad()
#     for step, batch in enumerate(tqdm_bar):
#         batch = batch.to(device)
#         neg_log_pxh, reg_term, mean_abs_z = model(batch, device)
#         loss = (neg_log_pxh + 0.001 * reg_term) / iter_size
#         loss.backward()
#         if model_config['clip_grad']:
#             grad_norm = gradient_clipping(model, gradnorm_queue)
#         else:
#             grad_norm = 0.
#         if (step + 1) % iter_size == 0:
#             optimizer.step()
#             optimizer.zero_grad()
#         if model_config['ema_decay'] > 0:
#             ema.update_model_average(model_ema, model)
#         nll_epoch.append(neg_log_pxh.item())
#         wandb.log({"Batch NLL": neg_log_pxh.item()}, commit=True) if not disable_tqdm and use_wandb else None
#     nll_epoch = np.mean(nll_epoch)
#     wandb.log({"Train Epoch NLL": nll_epoch}, commit=True) if not disable_tqdm and use_wandb else None
#     return nll_epoch


def train(model, loader, optimizer, device, model_config, disable_tqdm, model_ema, ema, gradnorm_queue=None, use_wandb=False):
    model.train()
    nll_epoch, n_samples = [], 0
    tqdm_bar = tqdm(loader, desc="Iteration", disable=disable_tqdm)
    for step, batch in enumerate(tqdm_bar):
        batch = batch.to(device)
        optimizer.zero_grad()
        neg_log_pxh, reg_term, mean_abs_z = model(batch, device)
        loss = neg_log_pxh + 0.001 * reg_term
        loss.backward()
        if model_config['clip_grad']:
            grad_norm = gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.
        optimizer.step()
        if model_config['ema_decay'] > 0:
            ema.update_model_average(model_ema, model)
        nll_epoch.append(neg_log_pxh.item())
        wandb.log({"Batch NLL": neg_log_pxh.item()}, commit=True) if not disable_tqdm and use_wandb else None
    nll_epoch = np.mean(nll_epoch)
    wandb.log({"Train Epoch NLL": nll_epoch}, commit=True) if not disable_tqdm and use_wandb else None
    return nll_epoch


def analyze(model, model_config, dataset_config, device, disable_tqdm, prop_dist=None):
    model.eval()
    n_samples, batch_size = model_config['n_samples'], model_config['sample_batch_size']
    batch_size = min(batch_size, n_samples)
    molecules = {'pos': [], 'onehot': [], 'node_mask': []}
    tqdm_bar = tqdm(range(int(n_samples / batch_size)), desc="Iteration", disable=disable_tqdm)
    for i in tqdm_bar:
        with torch.no_grad():
            pos, onehot, atom_num, degree, node_mask = model.module.sample(batch_size, dataset_config['max_n_nodes'],
                                                                   device, prop_dist)
            molecules['pos'].append(pos.detach().cpu())
            molecules['onehot'].append(onehot.detach().cpu())
            molecules['node_mask'].append(node_mask.detach().cpu())
    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity, rdkit_metrics, rdkit_unique = analyze_stability_for_molecules(molecules, dataset_config)
    return validity, rdkit_metrics, rdkit_unique


def valid(model, loader, device, disable_tqdm):
    model.eval()
    nll_epoch, n_samples = 0, 0
    tqdm_bar = tqdm(loader, desc="Iteration", disable=disable_tqdm)
    for step, batch in enumerate(tqdm_bar):
        batch = batch.to(device)
        with torch.no_grad():
            nll, _, _ = model(batch, device)
        nll_epoch += nll.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return nll_epoch / n_samples


def test(model, loader, device, disable_tqdm):
    model.eval()
    nll_epoch, n_samples = 0, 0
    tqdm_bar = tqdm(loader, desc="Iteration", disable=disable_tqdm)
    for step, batch in enumerate(tqdm_bar):
        batch = batch.to(device)
        with torch.no_grad():
            nll, _, _ = model(batch, device)
        nll_epoch += nll.item() * batch.num_graphs
        n_samples += batch.num_graphs
    return nll_epoch / n_samples


def main(rank, world_size, args):
    model_name = args.model
    dataset_name = args.data
    use_wandb = args.wandb
    model_config = yaml.load(open(os.path.join(CURRENT_PATH, 'config/model/' + model_name + '.yaml'), "r"),
                             Loader=yaml.FullLoader)
    dataset_config = yaml.load(open(os.path.join(CURRENT_PATH, 'config/dataset/' + dataset_name + ".yaml"), "r"),
                               Loader=yaml.FullLoader)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    init_seeds(model_config['seed'] + rank)

    # Initialization
    if torch.cuda.is_available():
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device('cpu')
    disable_tqdm = rank != 0
    if rank == 0 and use_wandb:
        wandb.init(config={**model_config, **dataset_config},
                   project=start_time + '_' + args.comment,
                   name=model_name + '_on_' + dataset_name)
        wandb.save('*.txt')
    # Prepare dataset.
    dataset_class = dataset_map[dataset_name]
    dataset = dataset_class(root=dataset_config['root'],
                            model_config=model_config,
                            dataset_config=dataset_config)
    print(len(dataset.data.x))
    split_idx = dataset.get_split_idx(len(dataset.data.n_nodes))

    if model_config['train_subset']:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx['train']))[:int(subset_ratio * len(split_idx['train']))]
        train_sampler = DistributedSampler(dataset[split_idx['train']][subset_idx], num_replicas=world_size,
                                           rank=rank, shuffle=True)
        train_loader = DataLoader(dataset[split_idx['train']][subset_idx], batch_size=model_config['batch_size'],
                                  shuffle=False, num_workers=model_config['num_workers'], sampler=train_sampler)
    else:
        train_sampler = DistributedSampler(dataset[split_idx['train']], num_replicas=world_size,
                                           rank=rank, shuffle=True)
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=model_config['batch_size'], shuffle=False,
                                  num_workers=model_config['num_workers'], sampler=train_sampler)
    if model_config['context']:
        prop_dist = DistributionProperty(train_loader, model_config['context_col'])
    else:
        prop_dist = None
    print("Lentgh of dataloader!", dist.get_rank(), len(train_loader))  # check all local_rank have equal batch
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=model_config['batch_size'] * 2,
                             shuffle=False, num_workers=model_config['num_workers'])
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=model_config['batch_size'] * 2,
                             shuffle=False, num_workers=model_config['num_workers'])
    if not disable_tqdm:
        print(f"Number of training samples: {len(dataset[split_idx['train']])}, "
              f"Number of validation samples: {len(dataset[split_idx['valid']])}, "
              f"Number of test samples: {len(dataset[split_idx['test']])}")
    del dataset

    model_class = model_map[model_name]
    model = model_class(model_config, dataset_config).to(device)
    model_config['ckpt_dir'] = "" if disable_tqdm else model_config['ckpt_dir']
    model_config['enable_tb'] = False if disable_tqdm else model_config['enable_tb']
    if model_config['load_ckpt']:
        ckpt = torch.load(model_config['load_ckpt_dir'])
        model.load_state_dict(ckpt['model_state_dict'])
        if not disable_tqdm:
            print('Load saved ckpt complete!')
    model = DistributedDataParallel(model, device_ids=[rank])
    model_wo_ddp = model.module
    num_params = sum(p.numel() for p in model_wo_ddp.parameters())
    print(f'Model successfully loaded, Number of Params: {num_params}') if not disable_tqdm else None
    if model_config['ema_decay'] > 0:
        model_ema = copy.deepcopy(model)
        ema = EMA(model_config['ema_decay'])
    else:
        model_ema = model
        ema = None
    if model_config['clip_grad']:
        gradnorm_queue = Queue()
        gradnorm_queue.add(3000)

    optimizer = optim.Adam(model.parameters(), lr=model_config['lr'], amsgrad=True, weight_decay=1e-12)
    if model_config['load_ckpt']:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    log_dir = os.path.join(model_config['log_dir'], model_name + '_' + dataset_name)
    if not os.path.exists(log_dir) and rank == 0:
        os.makedirs(log_dir)
    tb_writer = SummaryWriter(log_dir) if model_config['enable_tb'] else None
    log_path = os.path.join(log_dir, start_time + "log.txt")
    model_logs = {'model_config': model_config}
    data_logs = {'dataset_config': dataset_config}
    write_log(log_path, model_logs)
    write_log(log_path, data_logs)

    best_loss = 1e8
    best_mol_stable = 0.
    best_validity = 0.
    dist.barrier()
    print('Start training...') if not disable_tqdm else None

    for epoch in range(0, model_config['epochs']):
        train_loader.sampler.set_epoch(epoch)
        if not disable_tqdm:
            print("=====Epoch {}".format(epoch))
            print("Training...")
        train_loss = train(model, train_loader,  optimizer, device, model_config, disable_tqdm,
                           model_ema, ema, gradnorm_queue, use_wandb)
        if not disable_tqdm:
            write_log(log_path, f"\rEpoch: {epoch}, Loss {train_loss:.4f}")
            print(f"\rEpoch: {epoch}, Loss {train_loss:.4f}")

        if epoch % model_config['analyze_iter'] == 0 and epoch >= model_config['analyze_threshold']:
            print('Analyzing...') if not disable_tqdm else None
            validity, rdkit_metrics, rdkit_unique = analyze(model_ema, model_config, dataset_config,
                                                            device, disable_tqdm, prop_dist)
            validity, rdkit_metrics = validity.to(device), rdkit_metrics.to(device)

            validity_gather_list = [torch.zeros_like(validity) for _ in range(world_size)]
            rdkit_gather_list = [torch.zeros_like(rdkit_metrics) for _ in range(world_size)]
            dist.all_gather(validity_gather_list, validity)
            dist.all_gather(rdkit_gather_list, rdkit_metrics)
            validity = torch.cat(validity_gather_list, dim=0).mean(0) if not disable_tqdm else None
            rdkit_metrics = torch.cat(rdkit_gather_list, dim=0).mean(0) if not disable_tqdm else None
            analyze_dict = {
                'mol_stable': validity[0].item(),
                'atom_stable': validity[1].item(),
                'rdkit_validity': rdkit_metrics[0].item(),
                'rdkit_uniqueness': rdkit_metrics[1].item(),
                'rdkit_novelty': rdkit_metrics[2].item()
            } if not disable_tqdm else None
            if not disable_tqdm:
                wandb.log(analyze_dict) if use_wandb else None
                write_log(log_path, analyze_dict)

            print(analyze_dict) if not disable_tqdm else None
            if not disable_tqdm:
                if analyze_dict['mol_stable'] > best_mol_stable:
                    best_mol_stable = analyze_dict['mol_stable']
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model_wo_ddp.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                    }
                    torch.save(checkpoint, os.path.join(model_config['ckpt_dir'], f"{start_time}mol_stable.pt"))
                if analyze_dict['rdkit_validity'] > best_validity:
                    best_validity = analyze_dict['rdkit_validity']
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model_wo_ddp.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                    }
                    torch.save(checkpoint, os.path.join(model_config['ckpt_dir'], f"{start_time}validity.pt"))

        if not disable_tqdm and epoch % model_config['valid_iter'] == 0:
            print("Evaluating...")
            valid_loss = valid(model_ema, valid_loader, device, disable_tqdm)
            print(f"Epoch: {epoch}, Valid: {valid_loss:.4f}")
            write_log(log_path, f"Epoch: {epoch}, Valid: {valid_loss:.4f}")
            wandb.log({'Valid loss': valid_loss}, commit=True) if use_wandb else None
            if valid_loss < best_loss:
                best_loss = valid_loss
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model_wo_ddp.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # "scheduler_state_dict": scheduler.state_dict(),
                    "best_valid": best_loss,
                    'model_ema': model_ema.state_dict() if model_config['ema_decay'] > 0 else None
                }
                torch.save(checkpoint, os.path.join(model_config['ckpt_dir'], f"{start_time}checkpoint.pt"))

            if model_config['enable_tb']:
                tb_writer.add_scalar("evaluation/train", train_loss, epoch)
                tb_writer.add_scalar("evaluation/valid", valid_loss, epoch)

    if rank == 0:
        best_model = model
        best_ckpt = torch.load(os.path.join(model_config['ckpt_dir'], f"{start_time}checkpoint.pt"))
        print(f"Best valid: {best_ckpt['best_valid']:.4f}")
        write_log(log_path, f"Best valid: {best_ckpt['best_valid']:.4f}")
        best_model.module.load_state_dict(best_ckpt['model_state_dict'])

        test_loss = test(best_model, test_loader, device, disable_tqdm)
        print(f"Test: {test_loss:.4f}")
        wandb.log({'Test loss': test_loss}, commit=True) if use_wandb else None

    if model_config['enable_tb']:
        tb_writer.close()
    torch.distributed.destroy_process_group()
    print("Finished training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='gfmdiff', action='store',
                        help="molecular graph generation models")
    parser.add_argument("--data", type=str, default="qm9", action='store',
                        help="the training data")
    parser.add_argument("--wandb", action='store_true', default=False)
    parser.add_argument("--comment", type=str, default="None", action='store',
                        help="comment on the experiment")
    args, unknown = parser.parse_known_args()

    os.environ['NCCL_SHM_DISABLE'] = '1'
    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
