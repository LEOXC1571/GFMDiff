
import os
# Use only one GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
import yaml
import time
import argparse
import wandb
start_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())

import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from models import model_map
from data import dataset_map
from evaluator import analyze_stability_for_molecules
from utils import init_seeds, DistributionProperty, calc_prop_norm

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OUTPUT_PATH = os.path.join(CURRENT_PATH, 'output/sample')

warnings.filterwarnings('ignore')


def train_classifier(classifier, loader, optimizer, scheduler, criterion, model_config, mean, std, device):
    scheduler.step()
    classifier.train()
    loss_dict = {'loss': 0, 'loss_accum': []}
    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = classifier(batch)
        label = (batch.y[:, model_config['context_col']] - mean) / std
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        loss_dict['loss'] += loss.detach().cpu().item() * model_config['class_batch_size']
        loss_dict['loss_accum'].append(loss.detach().cpu().item())
    return loss_dict['loss'] / (step + 1)


def valid_classifier(classifier, loader, criterion, model_config, mean, std, device, gen_eval=False):
    classifier.eval()
    loss_dict = {'loss': 0, 'loss_accum': []}
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        label = batch.y[:, model_config['context_col']] if not gen_eval else batch.y[:, 0]
        with torch.no_grad():
            pred = classifier(batch, gen_eval=gen_eval)
            loss = criterion(std * pred + mean, label)
        loss_dict['loss'] += loss.cpu().item() * model_config['class_batch_size']
        loss_dict['loss_accum'].append(loss.cpu().item())
    return loss_dict['loss'] / (step + 1)


class DiffusionDataloader:
    def __init__(self, model, prop_dist, model_config, dataset_config, device, unkown_labels=False, batch_size=10, iterations=200):
        self.model = model
        self.batch_size = batch_size
        self.iterations = iterations
        self.prop_dist = prop_dist
        self.device = device
        self.unkown_labels = unkown_labels
        self.max_n_nodes = dataset_config['max_n_nodes']
        self.context_col = model_config['context_col'][0]
        self.mean = dataset_config['y_mean'][model_config['context_col'][0]]
        self.std = dataset_config['y_std'][model_config['context_col'][0]]
        self.i = 0

    def __iter__(self):
        return self

    def sample(self):
        pos, one_hot, atom_num, degree, node_mask, pair_mask, context = self.model.module.sample(self.batch_size, self.max_n_nodes, self.device, prop_dist=self.prop_dist)
        context = context * self.std + self.mean
        data = Data(x=one_hot.detach(), pos=pos.detach(), node_mask=node_mask.detach(),
                    pair_mask=pair_mask.detach(), y=context.detach())
        return data

    def __next__(self):
        if self.i <= self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations


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


def main_quality(rank, world_size, args):
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
    print(time.strftime('%m-%d-%H-%M-%S', time.localtime()))
    validity, rdkit_metrics, rdkit_unique = analyze(model, model_config, dataset_config, device,
                                                    n_samples=50/world_size, batch_size=50, rank=rank)
    print(time.strftime('%m-%d-%H-%M-%S', time.localtime()))
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


def main_quantity(rank, world_size, args):
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

    class_dir = args.class_dir
    classifier = model_map['egnn']().to(device)
    data = dataset_map['qm9'](root=dataset_config['root'], model_config=model_config, dataset_config=dataset_config)
    prop_split_idx = data.get_split_idx(len(data.data.n_nodes), task='prop')
    optimizer = optim.Adam(classifier.parameters(), lr=model_config['class_lr'], weight_decay=1e-12)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, model_config['class_epochs'])
    train_loader = DataLoader(data[prop_split_idx['train_prop']], batch_size=model_config['class_batch_size'],
                              shuffle=False, num_workers=model_config['num_workers'])
    valid_loader = DataLoader(data[prop_split_idx['valid']], batch_size=model_config['class_batch_size'] * 2,
                              shuffle=False, num_workers=model_config['num_workers'])
    test_loader = DataLoader(data[prop_split_idx['test']], batch_size=model_config['class_batch_size'] * 2,
                             shuffle=False, num_workers=model_config['num_workers'])
    mean, std = calc_prop_norm(data[prop_split_idx['valid']], model_config)
    criterion = nn.L1Loss()
    if os.path.exists(class_dir):
        classifier_ckpt = torch.load(class_dir)
        classifier.load_state_dict(classifier_ckpt['model_state_dict'])
    else:
        best_valid = 1e6
        prop_mean, prop_std = calc_prop_norm(train_loader.dataset, model_config)
        for epoch in range(model_config['class_epochs']):
            train_classifier(classifier, train_loader, optimizer, scheduler, criterion, model_config, prop_mean, prop_std, device)
            valid_loss = valid_classifier(classifier, valid_loader, criterion, model_config, prop_mean, prop_std, device)
            if valid_loss < best_valid:
                best_valid = valid_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_valid': valid_loss
                }
                torch.save(checkpoint, os.path.join(model_config['ckpt_dir'], f"{start_time}_prop_checkpoint.pt"))
        test_loss = valid_classifier(classifier, test_loader, criterion, model_config, prop_mean, prop_std, device)
        classifier_ckpt = torch.load(os.path.join(model_config['ckpt_dir'], f"{start_time}_prop_checkpoint.pt"))
        classifier.load_state_dict(classifier_ckpt['model_state_dict'])

    model_class = model_map[model_name]
    model = model_class(model_config, dataset_config).to(device)

    gen_ckpt = torch.load(args.ckpt_dir)
    model.load_state_dict(gen_ckpt['model_state_dict'])
    model = DistributedDataParallel(model, device_ids=[rank])
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model successfully loaded, Number of Params: {num_params}')
    prop_dist = DistributionProperty(train_loader, model_config['context_col'])
    prop_dist.set_normalizer(mean, std, model_config)
    sample_loader = DiffusionDataloader(model, prop_dist, model_config, dataset_config, batch_size=10, device=device)
    loss = valid_classifier(classifier, sample_loader, criterion, model_config, mean, std, device, gen_eval=True)
    print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='gfmdiff', action='store',
                        help="molecular graph generation models")
    parser.add_argument("--data", type=str, default="qm9", action='store',
                        help="the training data")
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--ckpt_dir", type=str, action='store')
    parser.add_argument("--class_dir", type=str, action='store')
    parser.add_argument("--task", type=str, default='quality', action='store')
    parser.add_argument('--wandb', action='store_true', default=False)
    args, unknown = parser.parse_known_args()
    os.environ['NCCL_SHM_DISABLE'] = '1'
    world_size = torch.cuda.device_count()
    if args.task == 'quality':
        mp.spawn(main_quality, args=(world_size, args), nprocs=world_size, join=True)
    elif args.task == 'quantity':
        mp.spawn(main_quantity, args=(world_size, args), nprocs=world_size, join=True)
    else:
        raise NotImplementedError('Unsupported task!')
