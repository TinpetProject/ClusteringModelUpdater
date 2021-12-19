from numpy import float32
from torch.utils.data import Dataset
import torch
import pandas as pd
from torch.utils.data.dataset import random_split

class PredictionDataset(Dataset):
    def __init__(self, data, transform=None, in_length=30, out_length=7):
        '''
        data - csv file
        transform - additional transformations
        in_length - length of input sequence
        out_length - length of output sequence
        '''
        self.data = pd.read_csv(data, header=0, index_col=0)

        columns = self.data.columns
        self.gt_columns = [col for col in columns if 'cal' in col]
        self.in_columns = [col for col in columns if col not in self.gt_columns]
        self.in_length = in_length
        self.out_length = out_length
        self.transform = transform

        self.length = len(self.data) - self.in_length - self.out_length
        self.X = self.data[self.in_columns].astype(float32).values
        self.y = self.data[self.gt_columns].astype(float32).values

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input, target = self.X[idx:idx+self.in_length], self.y[idx+self.in_length:idx+self.in_length+self.out_length]
        if self.transform:
            input = self.transform(input)
        return input, target

    def get_splits(self, test_ratio=.2):
        '''
        Returns train and test indices
        '''
        test_length = int(self.length * test_ratio)
        train_length = self.length - test_length

        return random_split(self, [train_length, test_length])

class CalibDataset(Dataset):
    def __init__(self, data, transform=None, frame_length=30):
        '''
        data - csv file
        transform - additional transformations
        frame_length - length of the frame data. Default 30
        '''
        self.data = pd.read_csv(data, header=0, index_col=0)
        columns = self.data.columns
        self.gt_columns = [col for col in columns if 'cal' in col]
        self.in_columns = [col for col in columns if col not in self.gt_columns]

        self.frame_length = frame_length
        self.transform = transform
        self.length = len(self.data) - self.frame_length
        self.X = self.data[self.in_columns].astype(float32).values
        self.y = self.data[self.gt_columns].astype(float32).values

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input, target = self.X[idx:idx+self.frame_length], self.y[idx:idx+self.frame_length]
        if self.transform:
            input = self.transform(input)

        return input, target

    def get_splits(self, test_ratio=.2):
        '''
        Returns train and test indices
        '''
        test_length = int(self.length * test_ratio)
        train_length = self.length - test_length

        return random_split(self, [train_length, test_length])

# class Loader:
#     def __init__(self, data, calib=False, batch_size=128, transform=None, in_length=30, out_length=7):
#         train_size = 

#         if calib:
#             self.dataset = CalibDataset(data, transform=transform, frame_length=in_length)
#         else:
#             self.dataset = PredictionDataset(data, transform=transform, in_length=in_length, out_length=out_length)
        

        