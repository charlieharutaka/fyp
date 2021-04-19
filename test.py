# import torch
# from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from utils.datasets import ChoralSingingDataset
from utils.transforms import BoxCoxTransform, ZScoreTransform
# from models.wavenet import WaveNet

# writer = SummaryWriter(log_dir="runs/model")
# model = WaveNet(cond_in_channels=64)
# nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Number of parameters: {nparams} | Receptive field: {model.receptive_field}")
# writer.add_graph(model, (torch.randn(32, 1, model.receptive_field + 399), torch.randn(32, 64, model.receptive_field + 399)))
# writer.close()

spectrogram_transform = nn.Sequential(BoxCoxTransform(0.05), ZScoreTransform())
csd = ChoralSingingDataset('data', 4093, n_mels=128, n_fft=800, spectrogram_transform=spectrogram_transform)
test = csd.original_data[0]
print(test[2].max(), test[2].min(), test[2].mean(), test[2].std())
# dataloader = DataLoader(csd, batch_size=1, shuffle=True)

# for batch in dataloader:
#     print(batch)
#     print(batch[2].unsqueeze(-1).shape)
#     break