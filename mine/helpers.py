import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MineDataset(Dataset):
    def __init__(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        
        self.x = x
        self.y = y
        assert len(x) == len(y), "Input and target data must contain same number of elements"
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

def dataloader(x, y, batch_size=1, shuffle=True):    
    dataset = MineDataset(x, y)
    
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    
    return dl
