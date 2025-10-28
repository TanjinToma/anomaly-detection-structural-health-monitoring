#!/usr/bin/env python3
# -*- coding: utf-8 -*-


## Helper to compute MSE per window
    
import torch
from torch.utils.data import DataLoader
import numpy as np

def window_mse_scores(X,ds,model,device):
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    scores = []
    with torch.no_grad():
        for x,_ in dl:
            x = x.to(device)
            yhat = model(x)
            # MSE per-sample
            err = ((yhat - x)**2).mean(dim=(1,2)).cpu().numpy()
            scores.extend(err.tolist())
    return np.array(scores)

