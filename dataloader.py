# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): 
        x = self.X[i]  # (T,F)
        return x, x    # AE: input == target

