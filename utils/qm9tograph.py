
import numpy as np
from rdkit import Chem

import torch
from torch_geometric.data import Data


types = ['H', 'C', 'N', 'O', 'F'],
allowable_features = {
    'atomic_num': [1, 6, 7, 8, 9],
    'formal_charge': [-1, 0, 1],
    'chirality': [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                  Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                  Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW],
    'numH': [0, 1, 2, 3, 4],
    'valence': [1, 2, 3, 4, 5],
    'degree': [1, 2, 3, 4, 5],

    'bond_type': [Chem.rdchem.BondType.SINGLE,
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC
                  ],
    'bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'bond_isconjugated': [False, True],
    'bond_inring': [False, True],
    'bond_stereo': ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]
}   # 27

atom_dic = [len(allowable_features['atomic_num']), len(allowable_features['formal_charge']),
            len(allowable_features['chirality']), len(allowable_features['numH']),
            len(allowable_features['valence']), len(allowable_features['degree'])]

atom_cumsum = np.cumsum(atom_dic)

fm_c = [-1, 0, 1]
degree = [1, 2, 3, 4, 5]
nh = [0, 1, 2, 3, 4]
chirality = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
             Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
hybrid = [Chem.rdchem.HybridizationType.UNSPECIFIED
          ]
iv = [0, 1, 2, 3]


def mol2graph(mol, target):
    atom_feature_list = []
    for atom in mol.GetAtoms():
        atom_feature = \
            [atom.GetAtomicNum()] + \
            [allowable_features['atomic_num'].index(atom.GetAtomicNum())] + \
            [atom.GetTotalValence()]
        atom_feature_list.append(atom_feature)

    edges_list = []
    edge_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_feature = [allowable_features['bond_type'].index(bond.GetBondType())]
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features_list),
                             dtype=torch.long)

    x = torch.tensor(np.array(atom_feature_list), dtype=torch.long)
    n_nodes = x.size(0)

    data = Data(x=x, y=target, n_nodes=n_nodes, edge_index=edge_index, edge_attr=edge_attr)
    return data
