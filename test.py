import torch
from torch.utils.data import DataLoader

from utils.datasets import ChoralSingingDataset

csd = ChoralSingingDataset('data', 4093)
dataloader = DataLoader(csd, batch_size=32, shuffle=True)
