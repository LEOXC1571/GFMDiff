# !/usr/bin/env python
# -*- Coding: utf-8 -*-

# @Filename: analyze.py
# @Author: Leo Xu
# @Date: 2022/12/5 15:48
# @Email: leoxc1571@163.com
# Description:

import torch
import numpy as np
from rdkit import Chem
from utils import get_bond_order, allowed_bonds, geom_predictor, BasicMolecularMetrics


def check_stability(pos, atom_type, dataset_config, debug=False):
    atom_decoder = dataset_config['atom_decoder']
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    nr_bonds = np.zeros(pos.shape[0], dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if 'qm9' in dataset_config['name']:
                order = get_bond_order(atom1, atom2, dist)
            elif dataset_config['name'] == 'drugs':
                order = geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)


def analyze_stability_for_molecules(molecule_list, dataset_config):
    one_hot = molecule_list['onehot']
    pos_pred = molecule_list['pos']
    node_mask = molecule_list['node_mask']

    n_nodes = node_mask.sum(1).long()
    n_samples = pos_pred.shape[0]

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    processed_list = []

    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()[0: n_nodes[i]]
        pos = pos_pred[i].cpu().detach()[0: n_nodes[i]]
        processed_list.append((pos, atom_type))

    for mol in processed_list:
        pos, atom_type = mol
        validity_results = check_stability(pos, atom_type, dataset_config)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_bonds / float(n_nodes.sum().item())

    metrics = BasicMolecularMetrics(dataset_config)
    rdkit_validity, rdkit_uniqueness, rdkit_novelty, rdkit_unique = metrics.evaluate(processed_list)

    validity = torch.tensor([[fraction_mol_stable, fraction_atm_stable]], device=one_hot.device)
    rdkit_metrics = torch.tensor([[rdkit_validity, rdkit_uniqueness, rdkit_novelty]], device=one_hot.device)
    return validity, rdkit_metrics, rdkit_unique
