import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from glycemicDataset import glycemicDataset

class run:
    def __init__(self):
        pass

    def run(self):
        
        train_dataset = glycemicDataset(X, y, sequence_length = 16)