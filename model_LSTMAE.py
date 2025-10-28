#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, feat_dim, hidden=64, latent=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=num_layers,
                               batch_first=True, dropout=0.0 if num_layers==1 else dropout)
        self.enc_to_latent = nn.Linear(hidden, latent)
        self.latent_to_dec = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=num_layers,
                               batch_first=True, dropout=0.0 if num_layers==1 else dropout)
        self.out = nn.Linear(hidden, feat_dim)

    def forward(self, x):
        # x: (B,T,F)
        B,T,F = x.shape
        enc_out, (h, c) = self.encoder(x)         # (B,T,H)
        # take last time step’s hidden
        h_last = enc_out[:,-1,:]                  # (B,H)
        z = self.enc_to_latent(h_last)            # (B,latent)
        h0 = torch.tanh(self.latent_to_dec(z)).unsqueeze(0)  # (1,B,H)
        c0 = torch.zeros_like(h0)                 # simple init
        # Teacher forcing with original input as decoder input (works well for AE)
        dec_out, _ = self.decoder(x, (h0, c0))    # (B,T,H)
        y = self.out(dec_out)                     # (B,T,F)
        return y