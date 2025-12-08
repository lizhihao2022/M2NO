import torch
from torch.utils.data import DataLoader, Dataset
import os.path as osp

from utils import UnitGaussianNormalizer, GaussianNormalizer


class ERA5Dataset:
    def __init__(self, data_path, raw_resolution=[512, 512, 80], 
                 sample_resolution=[512, 512, 20], eval_resolution=[512, 512, 20], 
                 in_t=1, out_t=1, duration_t=10, train_day=6, valid_day=2, test_day=2,
                 train_batchsize=10, eval_batchsize=10, 
                 normalize=True, normalizer_type='PGN', prop='temp', sub=False,
                 **kwargs):
        process_path = data_path.split('.')[0] + '_processed.pt'
        if osp.exists(process_path):
            print('Loading processed data from ', process_path)
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = torch.load(process_path)
        else:
            print('Processing raw data from ', data_path)
            data = torch.load(data_path)
            
            train_x, train_y, normalizer = self.pre_process(data[:train_day], mode='train', 
                                                in_t=in_t, out_t=out_t, duration_t=duration_t,
                                                normalize=normalize, normalizer_type=normalizer_type)
            valid_x, valid_y = self.pre_process(data[-test_day-valid_day:-test_day], mode='valid',
                                                in_t=in_t, out_t=out_t, duration_t=duration_t, 
                                                normalize=normalize, normalizer=normalizer)
            test_x, test_y = self.pre_process(data[-test_day:], mode='test', 
                                              in_t=in_t, out_t=out_t, duration_t=duration_t, 
                                              normalize=normalize, normalizer=normalizer)
            torch.save(((train_x, train_y), (valid_x, valid_y), (test_x, test_y)), process_path)
        
        if sub is not False:
            sub_index = int(len(train_x) * sub)
            train_x = train_x[:sub_index]
            train_y = train_y[:sub_index]
        
        self.train_dataset = ERA5Base(train_x, train_y, mode='train', prop=prop, raw_resolution=raw_resolution, sample_resolution=sample_resolution)
        self.valid_dataset = ERA5Base(valid_x, valid_y, mode='valid', prop=prop, raw_resolution=raw_resolution, sample_resolution=eval_resolution, )
        self.test_dataset = ERA5Base(test_x, test_y, mode='test', prop=prop, raw_resolution=raw_resolution, sample_resolution=eval_resolution, )
                
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)
    
    def pre_process(self, data, in_t, out_t, duration_t, mode='train', 
                    normalize=False, normalizer_type='PGN', normalizer=None, **kwargs):
        
        if mode == 'train':
            x = data[:, :in_t, :, :, :]
            y = data[:, in_t:in_t+1, :, :, :]
            for i in range(1, duration_t):
                x = torch.cat((x, data[:, i:in_t+i, :, :, :]), dim=0)
                y = torch.cat((y, data[:, in_t+i:in_t+i+1, :, :, :]), dim=0)
        else:
            x = data[:, out_t-in_t:out_t, :, :, :]
            y = data[:, out_t:out_t+1, :, :, :]
            for i in range(1, duration_t):
                x = torch.cat((x, data[:, out_t+i-in_t:out_t+i, :, :, :]), dim=0)
                y = torch.cat((y, data[:, out_t+i:out_t+i+1, :, :, :]), dim=0)
        
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

        x = x.squeeze(1)
        y = y.squeeze(1)
        
        grid_x = torch.linspace(-90, 90, x.shape[1])
        grid_x = grid_x.reshape(1, x.shape[1], 1, 1).repeat(x.shape[0], 1, x.shape[2], 1)
        grid_y = torch.linspace(-180, 180, x.shape[2])
        grid_y = grid_y.reshape(1, 1, x.shape[2], 1).repeat(x.shape[0], x.shape[1], 1, 1)
                
        x = torch.cat([x, grid_x, grid_y], dim=-1)
        
        if mode == 'train':
            return x, y, x_normalizer
        else:
            return x, y


class ERA5Base(Dataset):
    """
    A base class for the Navier-Stokes dataset.

    Args:
        x (list): The input data.
        y (list): The target data.
        mode (str, optional): The mode of the dataset. Defaults to 'train'.
        **kwargs: Additional keyword arguments.

    Attributes:
        mode (str): The mode of the dataset.
        x (list): The input data.
        y (list): The target data.
    """

    def __init__(self, x, y, mode='train', prop='temp', raw_resolution=[512, 512, 20], sample_resolution=[512, 512, 20], **kwargs):
        self.mode = mode
        sample_factor_0 = raw_resolution[0] // sample_resolution[0]
        sample_factor_1 = raw_resolution[1] // sample_resolution[1]
        
        grid = x[:, ::sample_factor_0, ::sample_factor_1, -2:]
        if prop == 'temp':
            self.x = x[:, ::sample_factor_0, ::sample_factor_1, :1]
            self.x = torch.cat([self.x, grid], dim=-1)
            self.y = y[:, ::sample_factor_0, ::sample_factor_1, :1]
        elif prop == 'wind_u':
            self.x = x[:, ::sample_factor_0, ::sample_factor_1, 1:2]
            self.x = torch.cat([self.x, grid], dim=-1)
            self.y = y[:, ::sample_factor_0, ::sample_factor_1, 1:2]
        elif prop == 'wind_v':
            self.x = x[:, ::sample_factor_0, ::sample_factor_1, 2:3]
            self.x = torch.cat([self.x, grid], dim=-1)
            self.y = y[:, ::sample_factor_0, ::sample_factor_1, 2:3]
        elif prop == 'vel':
            self.x = x[:, ::sample_factor_0, ::sample_factor_1, 3:]
            self.y = y[:, ::sample_factor_0, ::sample_factor_1, 3:]
        else:
            raise ValueError('Invalid property')
            
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
