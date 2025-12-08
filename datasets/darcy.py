import math
import torch
from torch.utils.data import Dataset, DataLoader
from h5py import File
import numpy as np

from utils import UnitGaussianNormalizer, GaussianNormalizer


class DarcyDataset:
    def __init__(self, data_path, raw_resolution=[512, 512],
                 sample_resolution=[256, 256], eval_resolution=[256, 256],
                 train_batchsize=8, eval_batchsize=8, train_ratio=0.8, 
                 valid_ratio=0.1, test_ratio=0.1, subset=False, subset_ratio=0.1, **kwargs):
        X, y = self.load_data(data_path)
        
        if subset:
            data_size = X.shape[0] * subset_ratio
        else:   
            data_size = X.shape[0]
        
        train_idx = int(data_size * train_ratio)
        valid_idx = int(data_size * (train_ratio + valid_ratio))
        test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
        
        train_x, train_y, normalizer = self.pre_process(X[:train_idx], y[:train_idx], raw_resolution, sample_resolution, mode='train', normalize=True, normalizer_type='PGN')
        valid_x, valid_y = self.pre_process(X[train_idx: valid_idx], y[train_idx: valid_idx], raw_resolution, eval_resolution, mode='valid', normalize=True, normalizer=normalizer)
        test_x, test_y = self.pre_process(X[valid_idx:test_idx], y[valid_idx:test_idx], raw_resolution, eval_resolution, mode='test', normalize=True, normalizer=normalizer)
        
        self.train_dataset = DarcyBase(train_x, train_y, mode='train')
        self.valid_dataset = DarcyBase(valid_x, valid_y, mode='valid')
        self.test_dataset = DarcyBase(test_x, test_y, mode='test')
        del X, y, train_x, train_y, valid_x, valid_y, test_x, test_y
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)
    
    def load_data(self, data_path):
        all_data = File(data_path, 'r')
        if 'coeff' in all_data.keys():
            x = torch.tensor(np.array(all_data['coeff']), dtype=torch.float32)
            y = torch.tensor(np.array(all_data['sol']), dtype=torch.float32)
        else:
            x = torch.tensor(np.array(all_data['nu']), dtype=torch.float32)
            y = torch.tensor(np.array(all_data['tensor']), dtype=torch.float32)
            y = y.view(y.shape[0], y.shape[2], y.shape[3])
        
        return x, y
    
    def pre_process(self, x, y, raw_resolution, sample_resolution, normalize=False, 
                    normalizer_type='PGN', mode='train', normalizer=None, **kwargs):
        sample_factor_0 = math.ceil(raw_resolution[0] / sample_resolution[0])
        sample_factor_1 = math.ceil(raw_resolution[1] / sample_resolution[1])

        x = x[:, ::sample_factor_0, ::sample_factor_1]
        y = y[:, ::sample_factor_0, ::sample_factor_1]
        
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1) 
        
        if normalize:
            if mode == 'train':
                if normalizer_type == 'PGN':
                    x_normalizer = UnitGaussianNormalizer(x)
                else:
                    x_normalizer = GaussianNormalizer(x)
                x = x_normalizer.encode(x)
            else:
                x = normalizer.encode(x)
        
        grid_x = torch.linspace(0, 1, x.shape[1])
        grid_x = grid_x.reshape(1, x.shape[1], 1, 1).repeat(x.shape[0], 1, x.shape[2], 1)
        grid_y = torch.linspace(0, 1, x.shape[2])
        grid_y = grid_y.reshape(1, 1, x.shape[2], 1).repeat(x.shape[0], x.shape[1], 1, 1)
        
        x = torch.cat([x, grid_x, grid_y], dim=-1)
        
        if normalize:
            if mode == 'train':
                return x, y, x_normalizer
            else:
                return x, y
        

class DarcyBase(Dataset):
    def __init__(self, x, y, mode='train', **kwargs):
        self.mode = mode
        self.x, self.y = x.contiguous(), y.contiguous()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
