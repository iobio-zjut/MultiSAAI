# from pyrosetta import *
# init(extra_options = "-constant_seed -mute all -read_only_ATOM_entries")

# Import necessary libraries
import numpy as np
import pickle
import torch
from dataProcessingUtils import *
import os
import pandas as pd
import sys

# from pyprotein import *
# import math
# import scipy
# import scipy.spatial
# import time
map = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
    'Z': 'GLU',
    'B': 'ASP',
    'X': 'UNK'
}

base_path = sys.argv[1]
fasta_dict1_path = os.path.join(base_path, "heavy.fasta")
fasta_dict2_path = os.path.join(base_path, "light.fasta")
fasta_dict3_path = os.path.join(base_path, "antigen.fasta")

def main():
    # fasta_dict1 = get_fasta_dict('./D42-2_20250605_105234/heavy.fasta')
    # fasta_dict2 = get_fasta_dict('./D42-2_20250605_105234/light.fasta')
    # fasta_dict3 = get_fasta_dict('./D42-2_20250605_105234/antigen.fasta')

    fasta_dict1 = get_fasta_dict(fasta_dict1_path)
    fasta_dict2 = get_fasta_dict(fasta_dict2_path)
    fasta_dict3 = get_fasta_dict(fasta_dict3_path)

    x0_1d = {}
    x1_1d = {}
    y1_1d = {}

    for key in fasta_dict1:
        x0_1d[key] = extract_AAs_properties_ab(transform(fasta_dict1[key]))
    # first_value = list(x0_1d.values())[0]
    # print(first_value)
    # first_value_shape = first_value.shape
    # print("shape:", first_value_shape)
    for key in fasta_dict2:
        x1_1d[key] = extract_AAs_properties_ab(transform(fasta_dict2[key]))
    print(1111)
    for key in fasta_dict3:
        y1_1d[key] = extract_AAs_properties_ag(transform(fasta_dict3[key]))
    # first_value = list(y1_1d.values())[0]
    # print(first_value)
    # first_value_shape = first_value.shape
    # print("shape:", first_value_shape)
    # output_dir = './D42-2_20250605_105234'
    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'H.pkl'), 'wb') as f:
        pickle.dump(x0_1d, f)

    with open(os.path.join(output_dir, 'L.pkl'), 'wb') as f:
        pickle.dump(x1_1d, f)

    with open(os.path.join(output_dir, 'antigen.pkl'), 'wb') as f:
        pickle.dump(y1_1d, f)




def extract_AAs_properties_ab(aas):

    _prop = np.zeros((20 + 24 + 1 + 7, len(aas)))
    for i in range(len(aas)):
        aa = aas[i]
        _prop[residuemap[aa], i] = 1
        _prop[20:44, i] = blosummap[aanamemap[aa]]
        # print('111111111',_prop[20:44, i])
        _prop[44, i] = min(i, len(aas) - i) * 1.0 / len(aas) * 2
        _prop[45:, i] = meiler_features[aa] / 5
    _prop = _prop.T
    prop_matrix_padded = zero_pad_prop_matrix(_prop, target_length=140)
    return prop_matrix_padded


def extract_AAs_properties_ag(aas):
    _prop = np.zeros((20 + 24 + 1 + 7, len(aas)))
    for i in range(len(aas)):
        aa = aas[i]
        _prop[residuemap[aa], i] = 1
        _prop[20:44, i] = blosummap[aanamemap[aa]]
        # print('111111111',_prop[20:44, i])
        _prop[44, i] = min(i, len(aas) - i) * 1.0 / len(aas) * 2
        _prop[45:, i] = meiler_features[aa] / 5
    _prop = _prop.T
    # prop_matrix_padded = zero_pad_prop_matrix(_prop, target_length=300)
    return _prop

def get_fasta_dict(txt):
    fasta_dict = {}  # 创建一个空字典用于存储FASTA数据

    with open(txt, 'r') as fasta_file:
        current_key = None
        current_sequence = []

        for line in fasta_file:
            line = line.strip()  # 去除行末的换行符和空白字符

            if line.startswith('>'):
                # 如果行以">"开头，说明这是一个标题行，作为字典的键
                if current_key is not None:
                    # 如果当前有一个键和序列，将它们存储在字典中
                    fasta_dict[current_key] = ''.join(current_sequence)

                # 提取标题行作为键，去掉">"
                current_key = line[1:]
                current_sequence = []  # 重置序列列表
            else:
                # 否则，将行添加到当前序列中
                current_sequence.append(line)

        # 处理文件末尾的最后一个键值对
        if current_key is not None:
            fasta_dict[current_key] = ''.join(current_sequence)
    return fasta_dict


def transform(sequence):
    seq = []
    for char in sequence:
        seq.append(map[char])
    return seq



def zero_pad_prop_matrix(matrix, target_length):
    num_vectors, vector_length = matrix.shape
    if num_vectors >= target_length:
        return matrix[:target_length, :]
    else:
        padded_matrix = np.zeros((target_length, vector_length))
        padded_matrix[:num_vectors, :] = matrix
        return padded_matrix


if __name__ == '__main__':
    main()



