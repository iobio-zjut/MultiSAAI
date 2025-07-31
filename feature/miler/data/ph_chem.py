import os
# import random
# from argparse import Namespace
# from collections.abc import MutableMapping
from typing import Any, Dict

#import matplotlib.pyplot as plt
import numpy as np
# import sklearn
import math
import torch
# from scipy.stats import norm
from torch.utils.tensorboard import SummaryWriter

aa_chars = np.array(list('AVLIPSTMEQHKRFYWDNCG'))


def seq_to_indices(seq, aa_chars='AVLIPSTMEQHKRFYWDNCG'):
    """
    :param seq: str,
    :param aa_chars: np.array,
    :return: np.array
    """
    seq_array = np.array(list(seq))
    indices = np.argmax(aa_chars[:, None] == seq_array, axis=0)
    return indices


HydrogenBondAcceptor_Donor = {
    'A': [5.0, 3.0],
    'V': [5.0, 3.0],
    'L': [5.0, 3.0],
    'I': [5.0, 3.0],
    'P': [5.0, 2.0],
    'S': [7.0, 4.0],
    'T': [7.0, 4.0],
    'M': [5.0, 3.0],
    'E': [9.0, 4.0],
    'Q': [8.0, 5.0],
    'H': [7.0, 4.0],
    'K': [6.0, 5.0],
    'R': [8.0, 7.0],
    'F': [5.0, 3.0],
    'Y': [7.0, 4.0],
    'W': [6.0, 4.0],
    'D': [9.0, 4.0],
    'N': [8.0, 5.0],
    'C': [5.0, 3.0],
    'G': [5.0, 3.0],
}

Charged_Side_Chains = {
    'A': [0.],
    'V': [0.],
    'L': [0.],
    'I': [0.],
    'P': [0.],
    'S': [0.],
    'T': [0.],
    'M': [0.],
    'E': [1.0],
    'Q': [0.],
    'H': [-1.0],
    'K': [-1.0],
    'R': [-1.0],
    'F': [0.],
    'Y': [0.],
    'W': [0.],
    'D': [1.0],
    'N': [0.],
    'C': [0.],
    'G': [0.],
}

Hydropathy_Index = {
    'A': [0.62],
    'V': [1.1],
    'L': [1.1],
    'I': [1.4],
    'P': [0.12],
    'S': [-0.18],
    'T': [-0.05],
    'M': [0.64],
    'E': [-0.74],
    'Q': [-0.85],
    'H': [-0.40],
    'K': [-1.5],
    'R': [-2.5],
    'F': [1.2],
    'Y': [0.26],
    'W': [0.81],
    'D': [-0.9],
    'N': [-0.78],
    'C': [0.29],
    'G': [0.48],
}

Volume = {
    'A': [88.6],
    'V': [140.0],
    'L': [166.7],
    'I': [166.7],
    'P': [112.7],
    'S': [89.0],
    'T': [116.1],
    'M': [162.9],
    'E': [138.4],
    'Q': [143.8],
    'H': [153.2],
    'K': [168.6],
    'R': [173.4],
    'F': [189.9],
    'Y': [193.6],
    'W': [227.8],
    'D': [111.1],
    'N': [114.1],
    'C': [108.5],
    'G': [60.1],
}
# Zamyatnin, A.A., Protein volume in solution, Prog. Biophys. Mol. Biol., 24:107-123 (1972), PMID: 4566650.

hydrogen_bond_acceptor_donor_values = np.array(list(HydrogenBondAcceptor_Donor.values()))
charged_side_chains_values = np.array([Charged_Side_Chains[aa][0] for aa in aa_chars])
hydropathy_index_values = np.array([Hydropathy_Index[aa][0] for aa in aa_chars])
volume_values = np.array([Volume[aa][0] for aa in aa_chars])


def update_matrix(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value


def generate_Chem_tensor(Abseq, Agseq, aa_chars=aa_chars,
                         hydrogen_bond_acceptor_donor_values=hydrogen_bond_acceptor_donor_values,
                         charged_side_chains_values=charged_side_chains_values,
                         hydropathy_index_values=hydropathy_index_values, volume_values=volume_values):
    """
    """
    Abseq_indices = seq_to_indices(Abseq, aa_chars)
    Agseq_indices = seq_to_indices(Agseq, aa_chars)

    tensor = np.zeros((len(Agseq), len(Abseq), 5))

    Ab_hydrogen_bond = hydrogen_bond_acceptor_donor_values[Abseq_indices]
    Ag_hydrogen_bond = hydrogen_bond_acceptor_donor_values[Agseq_indices]
    Ab_charged = charged_side_chains_values[Abseq_indices]
    Ag_charged = charged_side_chains_values[Agseq_indices]
    Ab_hydropathy = hydropathy_index_values[Abseq_indices]
    Ag_hydropathy = hydropathy_index_values[Agseq_indices]
    Ab_volume = volume_values[Abseq_indices]
    Ag_volume = volume_values[Agseq_indices]

    tensor[:, :, 0] = np.minimum(Ab_hydrogen_bond[:, 0][None, :], Ag_hydrogen_bond[:, 1][:, None])
    tensor[:, :, 1] = np.minimum(Ab_hydrogen_bond[:, 1][None, :], Ag_hydrogen_bond[:, 0][:, None])
    tensor[:, :, 2] = 0.5 * np.abs(Ab_charged[None, :] - Ag_charged[:, None])
    tensor[:, :, 3] = 1 - 0.25 * np.abs(Ab_hydropathy[None, :] - Ag_hydropathy[:, None])
    tensor[:, :, 4] = np.exp(-((Ab_volume[None, :] + Ag_volume[:, None]) - 282.52) ** 2 / (2 * (55.54 ** 2)))

    return torch.tensor(tensor, dtype=torch.float)

if __name__ == '__main__':
    # args = get_arguments()
    H_chain = 'QVQLVETGGGLIQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTFYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLERAGGMDVWGQGTMVTVSS'
    chain_length1 = len(H_chain)
    print("The length of the H chain is:", chain_length1)
    L_chain = 'EIVMTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSLYTFGQGTKVDIK'
    chain_length2 = len(L_chain)
    print("The length of the L chain is:", chain_length2)
    A_chain = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
    # Chem_matrix = generate_Chem_tensor(H_chain[i]+L_chain[i],A_chain[i])
    Chem_matrix_Hag = generate_Chem_tensor(H_chain, A_chain)
    Chem_matrix_Lag = generate_Chem_tensor(L_chain, A_chain)
    print("Chem_matrix", Chem_matrix_Hag)
    print("Chem_matrix",Chem_matrix_Hag.shape)