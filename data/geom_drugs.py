import os
import yaml
import random
import msgpack
from tqdm import tqdm
from rdkit import Chem
import torch.nn.functional as F
import numpy as np
from sklearn.utils import shuffle
import torch
from torch_geometric.data import InMemoryDataset, Data

from utils import drug2graph

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


class GEOM_Drugs(InMemoryDataset):
    def __init__(self,
                 root=None,
                 model_config=None,
                 dataset_config=None,
                 mol2graph=drug2graph,
                 transform=None,
                 pre_transform=None):
        self.root = root
        self.mol2graph = mol2graph
        self.remove_hs = False
        self.train_size = dataset_config['train_size']
        self.valid_size = dataset_config['valid_size']
        self.seed = model_config['seed']
        random.seed(self.seed)
        self.num_conf = dataset_config['num_conf']
        super(GEOM_Drugs, self).__init__(root, transform, pre_transform)
        self.transform, self.pre_transform = transform, pre_transform
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return ['drugs_crude.msgpack']

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return "geom_drugs_processed.pt"

    def download(self):
        if os.path.exists(os.path.join(self.raw_dir, self.processed_file_names[0])):
            return
        else:
            print("Please make sure raw files are downloaded")

    def process(self):
        print("Start preprocessing GEOM_Drugs datasets...")
        unpacker = msgpack.Unpacker(open(os.path.join(self.raw_dir, self.raw_file_names[0]), "rb"))
        smiles_list = []
        mol_id, conf_id = 0, 0
        data_list = []

        print("Start converting raw files to graphs...")
        for i, drugs_1k in enumerate(unpacker):
            print(f"Unpacking file {i}...")
            for smiles, all_info in tqdm(drugs_1k.items()):
                smiles_list.append(smiles)
                conformers = all_info['conformers']
                all_energies = []
                for conformer in conformers:
                    all_energies.append(conformer['totalenergy'])
                all_energies = np.array(all_energies)
                argsort = np.argsort(all_energies)
                lowest_energies = argsort[:self.num_conf]
                mol_id += 1
                for id in lowest_energies:
                    conformer = conformers[id]
                    conf_data = self.mol2graph(conformer, smiles)
                    data_list.append(conf_data)
                    conf_id += 1

        print("Total number of conformers saved", conf_id)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print(f"Num of valid molecules: {mol_id}, "
              f"Num of valid conformations: {conf_id}.")
        print(f"Saving data file to {self.processed_paths[0]}...")
        torch.save((data, slices), self.processed_paths[0])
        smiles_list = np.array(smiles_list)
        np.save(os.path.join(self.processed_dir, 'smiles_list.npy'), smiles_list)
        print("Data preprocessing finished!")

    def get_split_idx(self, data_size):
        ids = shuffle(range(data_size), random_state=self.seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:self.train_size]), torch.tensor(
            ids[self.train_size:self.train_size + self.valid_size]), torch.tensor(
            ids[self.train_size + self.valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict


if __name__ == "__main__":
    model_config = yaml.load(open(os.path.join(CURRENT_PATH, '../config/model/gfmdiff.yaml'), "r"),
                             Loader=yaml.FullLoader)
    dataset_config = yaml.load(open(os.path.join(CURRENT_PATH, '../config/dataset/drugs.yaml'), "r"),
                               Loader=yaml.FullLoader)
    root = '../molgen/geom_drugs'
    dataset = GEOM_Drugs(root, model_config, dataset_config)
