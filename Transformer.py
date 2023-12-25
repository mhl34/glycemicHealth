import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math
import numpy as np
import random
import datetime
import sys
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
from PositionalEncoding import PositionalEncoding
from hyperParams import hyperParams

class Transformer(nn.Module):
    # Constructor
    def __init__(
        self,
        num_tokens,
        embedding_dim_encode,
        embedding_dim_decode,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        norm_first,
        device):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.num_tokens = num_tokens
        self.embedding_dim_encode = embedding_dim_encode
        self.embedding_dim_decode = embedding_dim_decode
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p = dropout_p
        self.norm_first = norm_first
        self.device = device


        # LAYERS
        self.positional_encoder = PositionalEncoding(
            embedding_dim=embedding_dim_encode, dropout_p=dropout_p, max_len=5000, device=self.device
        )
        self.embedding = nn.Embedding(
            num_embeddings = self.num_tokens,
            embedding_dim = self.embedding_dim_encode
        )
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim_encode, 
            nhead=self.num_heads
        )
        self.transformerEncoder = nn.TransformerEncoder(
            self.transformerEncoderLayer, 
            num_layers=self.num_encoder_layers
        )
        self.transformerDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.embedding_dim_encode, 
            nhead=self.num_heads
        )
        self.transformerDecoder = nn.TransformerDecoder(
            self.transformerDecoderLayer, 
            num_layers=self.num_decoder_layers
        )
        self.transformer = nn.Transformer(
            d_model=self.embedding_dim_encode,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout_p,
            norm_first=self.norm_first,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embedding_dim_encode)
        self.fc1 = nn.Linear(self.embedding_dim_decode, self.num_tokens*4)
        self.fc2 = nn.Linear(self.num_tokens*4, self.num_tokens*2)
        self.fc3 = nn.Linear(self.num_tokens*2, self.num_tokens)

    # function: forward of model
    # input: src, tgt, tgt_mask
    # output: output after forward run through model
    def forward(self, src, tgt, tgt_mask=None):
        # Src size must be (batch_size, src, sequence_length)
        # Tgt size must be (batch_size, tgt, sequence_length)
        

        eda = src[0]
        hr = src[1]
        temp = src[2]
        
        eda = self.positional_encoder(self.embedding(eda) * math.sqrt(self.embedding_dim_encode))
        hr = self.positional_encoder(self.embedding(hr) * math.sqrt(self.embedding_dim_encode))
        temp = self.positional_encoder(self.embedding(temp) * math.sqrt(self.embedding_dim_encode))
        tgt = self.positional_encoder(self.embedding(tgt) * math.sqrt(self.embedding_dim_encode))

        # permute to have batch_size come first
        
        eda = eda.permute(1,0,2)
        hr = hr.permute(1,0,2)
        temp = temp.permute(1,0,2)
        tgt = tgt.permute(1,0,2)
        
        edaTransformerOut = self.transformer(eda, tgt, tgt_mask=tgt_mask)
        hrTransformerOut = self.transformer(hr, tgt, tgt_mask=tgt_mask)
        tempTransformerOut = self.transformer(temp, tgt, tgt_mask=tgt_mask)
        
        out = F.relu(self.fc1(torch.cat((edaTransformerOut, hrTransformerOut, tempTransformerOut), 2)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        return out

    # function: creates a mask with 0's in bottom left of matrix
    # input: size
    # output: mask
    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size,size) * float('-inf')).T
        for i in range(size):
            mask[i, i] = 0
        return mask