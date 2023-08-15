import os
import yaml
import random

from tqdm import tqdm
from rdkit import Chem

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from sklearn.utils import shuffle

from utils import mol2graph

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


class GEOM_QM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
        Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
        about 130,000 molecules with 19 regression targets.
        Each molecule includes complete spatial information for the single low
        energy conformation of the atoms in the molecule.
        In addition, we provide the atom features from the `"Neural Message
        Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | Target | Property                         | Description                                                                       | Unit                                        |
        +========+==================================+===================================================================================+=============================================+
        | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
        | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
        +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

        Args:
            root (string): Root directory where the dataset should be saved.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        Stats:
            .. list-table::
                :widths: 10 10 10 10 10
                :header-rows: 1

                * - #graphs
                  - #nodes
                  - #edges
                  - #features
                  - #tasks
                * - 130,831
                  - ~18.0
                  - ~37.3
                  - 11
                  - 19
        """

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self,
                 root=None,
                 model_config=None,
                 dataset_config=None,
                 mol2graph=mol2graph,
                 transform=None,
                 pre_transform=None):
        self.root = root
        self.mol2graph = mol2graph
        self.conditional = model_config['context']
        self.train_size = dataset_config['train_size']
        self.valid_size = dataset_config['valid_size']
        self.remove_hs = False
        super().__init__(root, transform, pre_transform)
        self.transform, self.pre_transform = transform, pre_transform
        self.seed = model_config['seed']
        random.seed(self.seed)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return "geom_qm9_processed.pt"

    def download(self):
        if os.path.exists(os.path.join(self.raw_dir, self.raw_file_names[0])):
            return
        else:
            try:
                import rdkit  # noqa
                file_path = download_url(self.raw_url, self.raw_dir)
                extract_zip(file_path, self.raw_dir)
                os.unlink(file_path)

                file_path = download_url(self.raw_url2, self.raw_dir)
                os.rename(os.path.join(self.raw_dir, '3195404'),
                          os.path.join(self.raw_dir, 'uncharacterized.txt'))
            except ImportError:
                path = download_url(self.processed_url, self.raw_dir)
                extract_zip(path, self.raw_dir)
                os.unlink(path)

    def process(self):
        print("Start converting raw files to graphs...")

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], sanitize=False, removeHs=self.remove_hs)
        smiles_list = []
        data_list = []

        for idx, mol in enumerate(tqdm(suppl)):
            if idx in skip:
                continue

            y = target[idx].unsqueeze(0)
            data = self.mol2graph(mol, y)
            n = mol.GetNumAtoms()

            pos = suppl.GetItemText(idx).split('\n')[4:4 + n]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            data.pos = pos
            assert len(data.x) == data.num_nodes
            data_list.append(data)

            smiles_list.append(mol2smiles(mol))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print(f"Num of valid molecules: {len(data_list)}, "
              f"num of valid conformations: {len(data_list)}, "
              f"num of bad cases {0}")
        print(f"Saving data file to {self.processed_paths[0]}...")
        torch.save((data, slices), self.processed_paths[0])
        smiles_list = np.array(smiles_list)
        np.save(os.path.join(self.processed_dir, 'smiles_list.npy'), smiles_list)

    def get_split_idx(self, data_size, task='gen'):
        ids = shuffle(range(data_size), random_state=self.seed)
        if not self.conditional:
            train_idx, val_idx, test_idx = torch.tensor(ids[:self.train_size]), torch.tensor(
                ids[self.train_size:self.train_size + self.valid_size]), torch.tensor(
                ids[self.train_size + self.valid_size:])
            split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        else:
            if task == 'gen':
                half_train_size = int(self.train_size / 2)
                train_idx, val_idx, test_idx = torch.tensor(ids[half_train_size:self.train_size]), torch.tensor(
                    ids[self.train_size:self.train_size + self.valid_size]), torch.tensor(
                    ids[self.train_size + self.valid_size:])
                split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
            elif task == 'prop':
                half_train_size = int(self.train_size / 2)
                train_prop_idx, train_gen_idx, val_idx, test_idx = torch.tensor(ids[:half_train_size]), torch.tensor(
                    ids[half_train_size:self.train_size]), torch.tensor(
                    ids[self.train_size:self.train_size + self.valid_size]), torch.tensor(
                    ids[self.train_size + self.valid_size:])
                split_dict = {'train_prop': train_prop_idx, 'train_gen': train_gen_idx,
                              'valid': val_idx, 'test': test_idx}
            else:
                raise ValueError("Unsupport task")
        return split_dict

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def retrive_context_prop(self, model_config):
        context_col = model_config['context_col']
        mean_ls, std_ls = [], []
        for col in context_col:
            mean_ls.append(self.mean(col))
            std_ls.append(self.std(col))
        return mean_ls, std_ls


if __name__ == "__main__":
    model_config = yaml.load(open(os.path.join(CURRENT_PATH, '../config/model/gfmdiff.yaml'), "r"),
                             Loader=yaml.FullLoader)
    dataset_config = yaml.load(open(os.path.join(CURRENT_PATH, '../config/dataset/qm9.yaml'), "r"),
                               Loader=yaml.FullLoader)
    root = '../data/molgen/geom_qm9'
    dataset = GEOM_QM9(root, model_config, dataset_config)
