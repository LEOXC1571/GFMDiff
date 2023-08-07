

import torch
from torch_geometric.data import Data
from utils import get_bond_order

allowable_features = {
    'atomic_num': [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83]
}
atom_decoder = ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']

allowable_max_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3, 'Si': 4, 'P': 5, 'S': 4, 'Cl': 1, 'As': 3,
                   'Br': 1, 'I': 1, 'Hg': 2, 'Bi': 5}

def drug2graph(conformer, smiles):
    coords = torch.tensor(conformer['xyz'], dtype=torch.float32)  # n x 4
    n_nodes = coords.shape[0]
    atom_num = coords[:, 0].long().unsqueeze(-1)
    pos = coords[:, 1:]
    atom_feature = [allowable_features['atomic_num'].index(i) for i in atom_num]
    pair_dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=-1)
    valencies = torch.zeros((n_nodes, 1), dtype=torch.int64)
    degree = torch.zeros((n_nodes, 1), dtype=torch.int64)
    for i in range(n_nodes):
        atom_i = atom_decoder[atom_feature[i]]
        for j in range(i+1, n_nodes):
            dist = pair_dist[i, j].item()
            atom_j = atom_decoder[atom_feature[j]]
            order = get_bond_order(atom_i, atom_j, dist, check_exists=True)
            is_bond = int(order > 0)
            degree[i] += order
            degree[j] += order
            valencies[i] += is_bond
            valencies[j] += is_bond
        possible_bonds = allowable_max_bonds[atom_i]
        if valencies[i] > possible_bonds:
            valencies[i] = possible_bonds
    atom_feature = torch.tensor(atom_feature).unsqueeze(-1)
    x = torch.cat([atom_num, atom_feature, valencies], dim=-1)
    data = Data(x=x, n_nodes=n_nodes, pos=pos)
    return data
