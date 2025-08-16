import math
import os
import torch
import scipy.io
from params import params_ad
from params import data_dir
from einops import rearrange
import numpy as np
import torch.nn as nn

class AD_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.all_paths = []
        with open(params_ad.fall_test_txt, 'r') as file:
            for line in file:
                self.all_paths.append(os.path.join(data_dir, line.strip()))#changed this to work on colab

        self.test_length = len(self.all_paths)

        self.filenames = []
        for i in range(self.test_length):
            self.filenames.append(self.all_paths[i])
            self.sigmoid = nn.Sigmoid()

    def __len__(self):
        return self.test_length

    def __getitem__(self, idx):
        fast_path = self.get_fast_path(idx)
        if os.path.exists(fast_path):
            data = torch.load(fast_path)
            return data
        else:
            filepath = self.filenames[idx]
            data = scipy.io.loadmat(filepath)['doppler_spectrum']
            data = torch.from_numpy(data).to(torch.float32)
            data = data[:, 5::10]
            data = rearrange(data, "a b -> b a")
            data_abs = data
            data_abs = self.normalize_abs(data_abs)
            torch.save(data_abs, fast_path)
            return data_abs
    
    def get_fast_path(self, idx):
        fast_dir = params_ad.test_fall_fast_path
        if not os.path.exists(fast_dir):
            os.mkdir(fast_dir)
        return fast_dir + str(idx) + ".pt"
    
    def normalize_abs(self, x):
        x_min = x.min()
        x_max = x.max()
        x = 2 * (x - x_min)/(x_max - x_min) - 1

        max_values, _ = torch.max(x, axis=-1)
        min_values, _ = torch.min(x, axis=-1)
        # columns_to_change = torch.where(max_values - min_values <= 1)[0]
        columns_to_change = torch.where(max_values<= 0.1)[0]
        x[columns_to_change] = -1

        x[x<-0.7] = -0.7
        x_min = x.min()
        x_max = x.max()
        x = 2 * (x - x_min)/(x_max - x_min) - 1
        return x

def from_path_eval(params = None, is_distributed=False):
    dataset = AD_Dataset()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False)


# ad_dataset = AD_Dataset()
# data = ad_dataset.__getitem__(0)
# print(data.shape)
