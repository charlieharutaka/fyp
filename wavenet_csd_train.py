import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from models.wavenet import WaveNet
from utils.datasets import ChoralSingingDataset
from utils.train import train_conditional_wavenet
from utils.transforms import BoxCoxTransform, ZScoreTransform, PowerToDecibelTransform

import torchaudio
if os.name == 'posix':
    torchaudio.set_audio_backend("sox_io")

CUDA_IS_AVAILABLE = torch.cuda.is_available()
print(f"Using {'CUDA' if CUDA_IS_AVAILABLE else 'CPU'}\nAudio Backend: {torchaudio.get_audio_backend()}")
if CUDA_IS_AVAILABLE:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Hyperparameters
wavenet_hp = {
    "layers": 10,
    "blocks": 4,
    "in_channels": 1,
    "cond_in_channels": 64,
    "cond_channels": 32,
    "dilation_channels": 32,
    "residual_channels": 32,
    "skip_channels": 256,
    "end_channels": 256,
    "classes": 256,
    "kernel_size": 2,
    "bias": False
}

BATCH_SIZE = 32
encoder = torchaudio.transforms.MuLawEncoding(wavenet_hp["classes"])
decoder = torchaudio.transforms.MuLawDecoding(wavenet_hp["classes"])

# The Model
model = WaveNet(**wavenet_hp)
model.to(device)
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"=====\nNumber of parameters: {nparams}\nReceptive field: {model.receptive_field}")

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# The Dataset
# We want to normalize the data using box-cox and z-score normalization
# spectrogram_transform = nn.Sequential(BoxCoxTransform(0.1), ZScoreTransform())
spectrogram_transform = PowerToDecibelTransform(torch.max)
dataset = ChoralSingingDataset('data', model.receptive_field, n_mels=64, n_fft=400, spectrogram_transform=spectrogram_transform)
# Calculate the splits
length_train = int(0.99 * len(dataset))
length_valid = len(dataset) - length_train

dataset_train, dataset_valid = random_split(dataset, [length_train, length_valid])
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False)

MODEL_NAME = "wavenet.csd3.p2d"

print(f"=====\nTraining Samples/Batches: {length_train}/{len(loader_train)}\nTesting Samples/Batches: {length_valid}/{len(loader_valid)}")
print(f"=====\nTraining {MODEL_NAME}...")

writer = SummaryWriter(log_dir=f"./runs/{MODEL_NAME}")
train_losses, valid_losses = train_conditional_wavenet(model, optimizer, criterion, 2, loader_train, loader_valid, encoder, print_every=1000, save_every=10000, validate_every=1000, save_as=MODEL_NAME, writer=writer, device=device)
torch.save(model.state_dict(), f"{MODEL_NAME}.final.pt")
torch.save(torch.tensor(train_losses), f"{MODEL_NAME}.train_losses.pt")
torch.save(torch.tensor(valid_losses), f"{MODEL_NAME}.valid_losses.pt")
writer.close()
