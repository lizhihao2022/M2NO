import torch
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import numpy as np
from h5py import File

from utils import UnitGaussianNormalizer, GaussianNormalizer


class NavierStokesDataset:
    def __init__(self, data_path, raw_resolution=[64, 64, 20], 
                 sample_resolution=[64, 64, 20], eval_resolution=[64, 64, 20], 
                 in_t=1, out_t=1, duration_t=10, 
                 train_batchsize=10, eval_batchsize=10, 
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, 
                 subset=False, subset_ratio=0.1, 
                 normalize=True, normalizer_type='PGN',
                 **kwargs):
        data, self.t = self.load_data(data_path)
        if subset:
            data_size = data.shape[0] * subset_ratio
        else:   
            data_size = data.shape[0]
        
        train_idx = int(data_size * train_ratio)
        valid_idx = int(data_size * (train_ratio + valid_ratio))
        test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
        
        train_x, train_y, normalizer = self.pre_process(data[:train_idx], mode='train',  # type: ignore
                                            raw_resolution=raw_resolution, sample_resolution=sample_resolution, 
                                            in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize, 
                                            normalizer_type=normalizer_type)
        valid_x, valid_y = self.pre_process(data[train_idx:valid_idx], mode='valid', # type: ignore
                                            raw_resolution=raw_resolution, sample_resolution=eval_resolution, 
                                            in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
                                            normalizer=normalizer)
        test_x, test_y = self.pre_process(data[valid_idx:test_idx], mode='test', # type: ignore
                                            raw_resolution=raw_resolution, sample_resolution=eval_resolution, 
                                            in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
                                            normalizer=normalizer)
        
        self.train_dataset = NavierStokesBase(train_x, train_y, mode='train')
        self.valid_dataset = NavierStokesBase(valid_x, valid_y, mode='valid')
        self.test_dataset = NavierStokesBase(test_x, test_y, mode='test')
        
        del train_x, train_y, valid_x, valid_y, test_x, test_y, data
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)
        
        self.vis_data = self.test_dataset[1000]

    def load_data(self, data_path):
        try:
            raw_data = sio.loadmat(data_path)
            u = raw_data['u']
            t = raw_data['t']
        except:
            raw_data = File(data_path, 'r')
            u = np.transpose(raw_data['u'], (3, 1, 2, 0))
            t = np.transpose(raw_data['t'], (1, 0))
            
        return u, t
    
    def pre_process(self, data, raw_resolution, sample_resolution, 
                    in_t, out_t, duration_t, mode='train', normalize=False, 
                    normalizer_type='PGN', normalizer=None, **kwargs):
        sample_factor_0 = raw_resolution[0] // sample_resolution[0]
        sample_factor_1 = raw_resolution[1] // sample_resolution[1]
        
        if mode == 'train':
            x = data[:, ::sample_factor_0, ::sample_factor_1, :in_t]
            y = data[:, ::sample_factor_0, ::sample_factor_1, in_t:in_t+1]
            for i in range(1, duration_t):
                x = np.concatenate((x, data[:, ::sample_factor_0, ::sample_factor_1, i:in_t+i]), axis=0)
                y = np.concatenate((y, data[:, ::sample_factor_0, ::sample_factor_1, in_t+i:in_t+i+1]), axis=0)
        else:
            x = data[:, ::sample_factor_0, ::sample_factor_1, out_t-in_t:out_t]
            y = data[:, ::sample_factor_0, ::sample_factor_1, out_t:out_t+1]
            for i in range(1, duration_t):
                x = np.concatenate((x, data[:, ::sample_factor_0, ::sample_factor_1, out_t+i-in_t:out_t+i]), axis=0)
                y = np.concatenate((y, data[:, ::sample_factor_0, ::sample_factor_1, out_t+i:out_t+i+1]), axis=0)
        
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
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
                
        grid_x = torch.linspace(0, 1, x.shape[1])
        grid_x = grid_x.reshape(1, x.shape[1], 1, 1).repeat(x.shape[0], 1, x.shape[2], 1)
        grid_y = torch.linspace(0, 1, x.shape[2])
        grid_y = grid_y.reshape(1, 1, x.shape[2], 1).repeat(x.shape[0], x.shape[1], 1, 1)
        
        x = torch.cat([x, grid_x, grid_y], dim=-1)

        if mode == 'train':
            return x, y, x_normalizer
        else:
            return x, y


class NavierStokesBase(Dataset):
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

    def __init__(self, x, y, mode='train', **kwargs):
        self.mode = mode
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
