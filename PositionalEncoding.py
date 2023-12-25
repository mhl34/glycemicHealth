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

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout_p, max_len, device):
        super().__init__()

        # Layers
        self.dropout = nn.Dropout(dropout_p)
        self.positional_encoding = self.get_positional_encoding(max_len, embedding_dim).to(device)

    def get_positional_encoding(self, max_len, embedding_dim):
        positional_encoding = torch.zeros(max_len, embedding_dim)
        # column of positions
        position = torch.arange(0, max_len, dtype=torch.float32).view(-1,1)
        # division term
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))
        # even numbered positional encoding
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # odd numbered positional encoding
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0,1)
        return positional_encoding

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.positional_encoding[:token_embedding.size(0), :])