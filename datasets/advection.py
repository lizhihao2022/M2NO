import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from h5py import File

from utils import UnitGaussianNormalizer, GaussianNormalizer


class AdvectionDataset:
    def __init__(self, data_path, raw_resolution=[1024, 201], 
                 sample_resolution=[128, 1], eval_resolution=[128, 1], 
                 in_t=1, out_t=1, duration_t=10, start_x=0, end_x=1,
                 train_batchsize=10, eval_batchsize=10, 
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, 
                 subset=False, subset_ratio=0.1, 
                 normalize=True, normalizer_type='PGN',
                 **kwargs):
        data = self.load_data(data_path)
        if subset:
            data_size = data.shape[0] * subset_ratio
        else:   
            data_size = data.shape[0]
        
        self.start_x = start_x
        self.end_x = end_x
        
        train_idx = int(data_size * train_ratio)
        valid_idx = int(data_size * (train_ratio + valid_ratio))
        test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
        
        train_x, train_y, normalizer = self.pre_process(data[:train_idx], mode='train', 
                                            raw_resolution=raw_resolution, sample_resolution=sample_resolution, 
                                            in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize, 
                                            normalizer_type=normalizer_type)
        valid_x, valid_y = self.pre_process(data[train_idx:valid_idx], mode='valid',
                                            raw_resolution=raw_resolution, sample_resolution=eval_resolution, 
                                            in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
                                            normalizer=normalizer)
        test_x, test_y = self.pre_process(data[valid_idx:test_idx], mode='test',
                                            raw_resolution=raw_resolution, sample_resolution=eval_resolution, 
                                            in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
                                            normalizer=normalizer)
        
        self.train_dataset = AdvectionBase(train_x, train_y, mode='train')
        self.valid_dataset = AdvectionBase(valid_x, valid_y, mode='valid')
        self.test_dataset = AdvectionBase(test_x, test_y, mode='test')
        
        del train_x, train_y, valid_x, valid_y, test_x, test_y, data
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)

    def load_data(self, data_path):
        f = File(data_path, 'r')
        u = torch.tensor(np.array(f['tensor']), dtype=torch.float32)
        
        return u
    
    def pre_process(self, data, raw_resolution, sample_resolution, 
                    in_t, out_t, duration_t, mode='train', normalize=False, 
                    normalizer_type='PGN', normalizer=None, **kwargs):
        sample_factor_x = raw_resolution[0] // sample_resolution[0]
        
        if mode == 'train':
            x = data[:, :in_t, ::sample_factor_x]
            y = data[:, in_t:in_t+1, ::sample_factor_x]
            for i in range(1, duration_t):
                x = torch.concatenate((x, data[:, i:in_t+i, ::sample_factor_x]), axis=0)
                y = torch.concatenate((y, data[:, in_t+i:in_t+i+1, ::sample_factor_x]), axis=0)
        else:
            x = data[:, out_t-in_t:out_t, ::sample_factor_x]
            y = data[:, out_t:out_t+1, ::sample_factor_x]
            for i in range(1, duration_t):
                x = torch.concatenate((x, data[:, out_t+i-in_t:out_t+i, ::sample_factor_x]), axis=0)
                y = torch.concatenate((y, data[:, out_t+i:out_t+i+1, ::sample_factor_x]), axis=0)
        
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        y = y.reshape(y.shape[0], y.shape[2], y.shape[1])
        
        if normalize:
            if mode == 'train':
                if normalizer_type == 'PGN':
                    x_normalizer = UnitGaussianNormalizer(x)
                else:
                    x_normalizer = GaussianNormalizer(x)
                x = x_normalizer.encode(x)
            else:
                x = normalizer.encode(x)
        else:
            x_normalizer = None
                
        grid_x = torch.linspace(self.start_x, self.end_x, x.shape[1])
        grid_x = grid_x.reshape(1, x.shape[1], 1).repeat(x.shape[0], 1, 1)

        x = torch.cat([x, grid_x], dim=-1)

        if mode == 'train':
            return x, y, x_normalizer
        else:
            return x, y


class AdvectionBase(Dataset):
    def __init__(self, x, y, mode='train', **kwargs):
        self.mode = mode
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
