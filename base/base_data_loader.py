# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler,SequentialSampler
import torch
from torch.utils.data import Sampler

class CustomSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, seed, shuffle, validation_split, 
                 test_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.seed = seed

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler, self.test_sampler = self._split_sampler()

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self):
        idx_full = np.arange(self.n_samples)

        #np.random.seed(self.seed)
        #np.random.shuffle(idx_full)
        
        if isinstance(self.validation_split, int) or isinstance(self.test_split, int):
            assert self.validation_split > 0 or self.test_split > 0
            assert self.validation_split < self.n_samples or self.test_split < self.n_samples, \
                "validation set size or test set size is configured to be larger than entire dataset."
            len_valid = self.validation_split
            len_test  = self.test_split
        else:
            len_valid = int(self.n_samples * self.validation_split)
            len_test  = int(self.n_samples * self.test_split)
        len_train =0
        len_train = len(idx_full)  - len_valid - len_test
        train_idx = idx_full[0:int(len_train)]
        print("train_idx",train_idx)
        valid_idx  = idx_full[len_train: (len_valid+len_train)]
        print("valid_idx", valid_idx)
        test_idx = np.delete(idx_full, np.arange(0, len_valid+len_train))
        print("test_idx", test_idx)

        train_sampler = CustomSampler(train_idx)
        valid_sampler = CustomSampler(valid_idx)
        test_sampler = CustomSampler(test_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)
        # valid_data = self.pair_df.iloc[valid_idx]  # 使用 train_idx
        # print("valid data:", valid_data)
        return train_sampler, valid_sampler, test_sampler

    def split_dataset(self, valid=False, test=False):
        if valid:
            assert len(self.valid_sampler) != 0, "validation set size ratio is not positive"
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        if test:
            assert len(self.test_sampler) != 0, "test set size ratio is not positive"
            return DataLoader(sampler=self.test_sampler, **self.init_kwargs)
    def eval_dataest(self, predict=False):
        if predict:
            assert len(self.test_sampler) != 0, "test set size ratio is not positive"
            return DataLoader(sampler=self.test_sampler, **self.init_kwargs)