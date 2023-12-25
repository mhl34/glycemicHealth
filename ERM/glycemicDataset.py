from torch.utils.data import Dataset

class glycemicDataset(Dataset):
    def __init__(self, X, y, sequence_length = 16):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return self.X.__len__() - self.sequence_length + 1
    
    def __getitem__(self, index):
        return (self.X[index : index + self.sequence_length], self.y[index])