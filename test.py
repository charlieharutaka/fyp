# import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.datasets import ChoralSingingDataset
# from models.wavenet import WaveNet

# writer = SummaryWriter(log_dir="runs/model")
# model = WaveNet(cond_in_channels=64)
# nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Number of parameters: {nparams} | Receptive field: {model.receptive_field}")
# writer.add_graph(model, (torch.randn(32, 1, model.receptive_field + 399), torch.randn(32, 64, model.receptive_field + 399)))
# writer.close()

csd = ChoralSingingDataset('data', 4093, segment_size=32)
print(csd.cumulative_lengths)
print(csd[0])
print(csd[0][0].shape, csd[0][1].shape, csd[0][2].shape)
# dataloader = DataLoader(csd, batch_size=1, shuffle=True)

# for batch in dataloader:
#     print(batch)
#     print(batch[2].unsqueeze(-1).shape)
#     break