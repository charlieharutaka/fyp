from torch.utils import data
from utils.datasets import VocalSetDataset
from pprint import pprint

import torch

dataset = VocalSetDataset()
print(len(dataset))
print(dataset[0])