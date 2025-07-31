# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join, exists
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from bert_data_prepare.tokenizer import get_tokenizer
from base import BaseDataLoader
from bert_data_prepare.utility import is_valid_aaseq
from transformers import AutoTokenizer


class Antibody_Antigen_Dataset_AbDab(BaseDataLoader):
    def __init__(self, logger,
                 seed,
                 batch_size,
                 validation_split,
                 test_split,
                 num_workers,
                 data_dir,
                 antibody_vocab_dir,
                 antibody_tokenizer_dir,
                 tokenizer_name='common',
                 # receptor_tokenizer_name='common',
                 token_length_list='2,3',
                 # receptor_token_length_list='2,3',
                 antigen_seq_name='antigen',
                 heavy_seq_name='Heavy',
                 light_seq_name='Light',
                 label_name='Label',
                 # receptor_seq_name='beta',
                 test_antibodys=100,
                 # neg_ratio=1.0,
                 shuffle=False,
                 antigen_max_len=None,
                 heavy_max_len=None,
                 light_max_len=None, ):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        self.heavy_seq_name = heavy_seq_name
        self.light_seq_name = light_seq_name
        self.antigen_seq_name = antigen_seq_name
        self.label_name = label_name

        self.test_antibodys = test_antibodys
        self.shuffle = shuffle
        self.heavy_max_len = heavy_max_len
        self.light_max_len = light_max_len
        self.antigen_max_len = antigen_max_len

        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()

        self.HeavyTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                            add_hyphen=False,
                                            logger=self.logger,
                                            vocab_dir=antibody_vocab_dir,
                                            token_length_list=token_length_list)
        self.heavy_tokenizer = self.HeavyTokenizer.get_bert_tokenizer(
            max_len=self.heavy_max_len,
            tokenizer_dir=antibody_tokenizer_dir)

        self.LightTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                            add_hyphen=False,
                                            logger=self.logger,
                                            vocab_dir=antibody_vocab_dir,
                                            token_length_list=token_length_list)
        self.light_tokenizer = self.LightTokenizer.get_bert_tokenizer(
            max_len=self.light_max_len,
            tokenizer_dir=antibody_tokenizer_dir)

        self.AntigenTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=antibody_vocab_dir,
                                              token_length_list=token_length_list)
        self.antigen_tokenizer = self.AntigenTokenizer.get_bert_tokenizer(
            max_len=self.antigen_max_len,
            tokenizer_dir=antibody_tokenizer_dir)

        esm_dir = '/home/data/user/lvzexin/zexinl/MultiSAAI/checkpoint/esm2/esm2_150m'

        self.antigen_tokenizer = AutoTokenizer.from_pretrained(esm_dir,
                                                               cache_dir="/home/data/user/lvzexin/zexinl/MultiSAAI/checkpoint/esm2/esm2_150m/cache",
                                                               max_len=self.antigen_max_len)

        dataset = self._get_dataset(pair_df=self.pair_df)
        super().__init__(dataset, batch_size, seed, shuffle, validation_split, test_split,
                         num_workers)

    def get_heavy_tokenizer(self):
        return self.heavy_tokenizer

    def get_light_tokenizer(self):
        return self.light_tokenizer

    def get_antibody_tokenizer(self):
        return self.heavy_tokenizer

    def get_antigen_tokenizer(self):
        return self.antigen_tokenizer

    def get_test_dataloader(self):
        return self.test_dataloader

    def _get_dataset(self, pair_df):
        # print(pair_df.columns)
        abag_dataset = AbAGDataset_CovAbDab(
            heavy_seqs=list(pair_df[self.heavy_seq_name]),
            light_seqs=list(pair_df[self.light_seq_name]),
            antigen_seqs=list(pair_df[self.antigen_seq_name]),
            labels=list(pair_df[self.label_name]),
            antibody_split_fun=self.HeavyTokenizer.split,
            antigen_split_fun=self.AntigenTokenizer.split,
            antibody_tokenizer=self.heavy_tokenizer,
            antigen_tokenizer=self.antigen_tokenizer,
            antibody_max_len=self.heavy_max_len,
            antigen_max_len=self.antigen_max_len,
            logger=self.logger
        )
        # print("get_dataset",len(heavy_seqs))
        return abag_dataset

    # def _split_dataset(self):
    # if exists(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv')):
    #     test_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv'))
    #     self.logger.info(f'Loading created unseen antibodys for test with shape {test_pair_df.shape}')

    # antibody_list = list(set(self.pair_df['antibody']))
    # selected_antibody_index_list = self.rng.integers(len(antibody_list), size=self.test_antibodys)
    # self.logger.info(f'Select {self.test_antibodys} from {len(antibody_list)} antibody')
    # selected_antibodys = [antibody_list[i] for i in selected_antibody_index_list]
    # test_pair_df = self.pair_df[self.pair_df['antibody'].isin(selected_antibodys)]
    # test_pair_df.to_csv(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv'), index=False)

    # selected_antibodys = list(set(test_pair_df['antibody']))
    # print("split_dataset")
    # train_valid_pair_df = self.pair_df[~self.pair_df['antibody'].isin(selected_antibodys)]
    #
    # self.logger.info(f'{len(train_valid_pair_df)} pairs for train and valid and {len(test_pair_df)} pairs for test.')
    #
    # return train_valid_pair_df, test_pair_df

    def _create_pair(self):
        pair_df = pd.read_csv(self.data_dir)
        # pair_df = pd.read_csv(self.data_dir, encoding='utf-8',errors='ignore')
        #
        with open(self.data_dir, 'r', encoding='utf-8', errors='ignore') as f:
            pair_df = pd.read_csv(f)
        if self.shuffle:
            pair_df = pair_df.sample(frac=1).reset_index(drop=True)
            self.logger.info("Shuffling dataset")
        self.logger.info(f"There are {len(pair_df)} samples")

        return pair_df

    def _load_seq_pairs(self):
        self.logger.info(f'Loading from {self.using_dataset}...')
        self.logger.info(f'Loading {self.antibody_seq_name} and {self.receptor_seq_name}')
        column_map_dict = {'alpha': 'cdr3a', 'beta': 'cdr3b', 'antibody': 'antibody'}
        keep_columns = [column_map_dict[c] for c in [self.antibody_seq_name, self.receptor_seq_name]]

        df_list = []
        for dataset in self.using_dataset:
            df = pd.read_csv(join(self.data_dir, dataset, 'full.csv'))
            df = df[keep_columns]
            df = df[(df[keep_columns[0]].map(is_valid_aaseq)) & (df[keep_columns[1]].map(is_valid_aaseq))]
            self.logger.info(f'Loading {len(df)} pairs from {dataset}')
            df_list.append(df[keep_columns])
        df = pd.concat(df_list)
        self.logger.info(f'Current data shape {df.shape}')
        df_filter = df.dropna()
        df_filter = df_filter.drop_duplicates()
        self.logger.info(f'After dropping na and duplicates, current data shape {df_filter.shape}')

        column_rename_dict = {column_map_dict[c]: c for c in [self.antibody_seq_name, self.receptor_seq_name]}
        df_filter.rename(columns=column_rename_dict, inplace=True)

        df_filter['label'] = [1] * len(df_filter)
        df_filter.to_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'), index=False)
        print("load_seq_pair")
        return df_filter


class AbAGDataset_CovAbDab(Dataset):
    def __init__(self, heavy_seqs,
                 light_seqs,
                 antigen_seqs,
                 labels,
                 antibody_split_fun,
                 antigen_split_fun,
                 antibody_tokenizer,
                 antigen_tokenizer,
                 antibody_max_len,
                 antigen_max_len,
                 logger):
        self.heavy_seqs = heavy_seqs
        self.light_seqs = light_seqs
        self.antigen_seqs = antigen_seqs
        self.labels = labels
        self.antibody_split_fun = antibody_split_fun
        self.antigen_split_fun = antigen_split_fun
        self.antibody_tokenizer = antibody_tokenizer
        self.antigen_tokenizer = antigen_tokenizer
        self.antibody_max_len = antibody_max_len
        self.antigen_max_len = antigen_max_len
        self.logger = logger
        self._has_logged_example = False

    def __len__(self):
        print("len(self.heavy_seqs)", len(self.heavy_seqs))
        return len(self.labels)

    def __getitem__(self, i):
        heavy, light, antigen = self.heavy_seqs[i], self.light_seqs[i], self.antigen_seqs[i]
        # print("i=",i)
        # print("heavy_seqs",self.heavy_seqs[i])
        # print("light_seqs", self.light_seqs[i])
        label = self.labels[i]
        heavy_tensor = self.antibody_tokenizer(self._insert_whitespace(self.antibody_split_fun(heavy)),
                                               padding="max_length",
                                               max_length=self.antibody_max_len,
                                               truncation=True,
                                               return_tensors="pt")
        light_tensor = self.antibody_tokenizer(self._insert_whitespace(self.antibody_split_fun(light)),
                                               padding="max_length",
                                               max_length=self.antibody_max_len,
                                               truncation=True,
                                               return_tensors="pt")

        antigen_tensor = self.antigen_tokenizer(antigen,
                                                padding="max_length",
                                                max_length=self.antigen_max_len,
                                                truncation=True,
                                                return_tensors="pt")

        # label_tensor = torch.FloatTensor(np.atleast_1d(label))

        heavy_tensor = {k: torch.squeeze(v) for k, v in heavy_tensor.items()}
        light_tensor = {k: torch.squeeze(v) for k, v in light_tensor.items()}
        antigen_tensor = {k: torch.squeeze(v) for k, v in antigen_tensor.items()}
        return heavy_tensor, light_tensor, antigen_tensor

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)