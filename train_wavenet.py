import os

import torch
from torch import cuda as torch_cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchaudio
if os.name == 'posix':
    torchaudio.set_audio_backend("sox_io")

from models.wavenet import WaveNet
from utils.datasets import SegmentedAudioDataset

CUDA_IS_AVAILABLE = torch_cuda.is_available()
print(f"Using {'CUDA' if CUDA_IS_AVAILABLE else 'CPU'} | Audio Backend: {torchaudio.get_audio_backend()}")
if CUDA_IS_AVAILABLE:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

"""
Initialize Model
"""

model = WaveNet(layers=14,
                blocks=2,
                dilation_channels=32,
                residual_channels=32,
                skip_channels=256,
                end_channels=256,
                classes=256,
                bias=True)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model.to(device)
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {nparams} | Receptive field: {model.receptive_field}")

"""
Datasets
"""

DATA_DIR = './data/'

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
arctic = torchaudio.datasets.CMUARCTIC(DATA_DIR, download=True)
audio_data = [data[0] for data in arctic]
dataset = SegmentedAudioDataset(audio_data, model.receptive_field)
length_train = int(0.9 * len(dataset))
length_test = len(dataset) - length_train
dataset_train, dataset_test = random_split(dataset, [length_train, length_test])

BATCH_SIZE = 8

loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training Samples/Batches: {length_train}/{len(loader_train)} | Testing Samples/Batches: {length_test}/{len(loader_test)}")

"""
Training
"""

PRINT_EVERY = 500
SAVE_EVERY = 10000
MODEL_NAME = "wavenet.segmented"

encoder = torchaudio.transforms.MuLawEncoding(256)
decoder = torchaudio.transforms.MuLawDecoding(256)

def train(epochs, epochs_start=0, verbose=True):
    counter = 1
    train_losses = []
    running_loss = 0.0

    for epoch in range(epochs_start, epochs_start + epochs):
        model.train()
        for t, batch in enumerate(loader_train):
            optimizer.zero_grad()

            inputs, target = batch
            target = encoder(target)
            inputs = inputs.to(device).unsqueeze(1)
            target = target.to(device)
            # Get prediction
            preds = model(inputs)
            # Calculate loss
            loss = F.cross_entropy(preds, target)
            loss.backward()
            optimizer.step()
            # Stats
            train_losses.append(loss.item())
            running_loss += loss.item()

            if counter % PRINT_EVERY == 0:
                print(f'Epoch: {epoch+1} | Iteration: {t+1} | Loss: {running_loss / PRINT_EVERY}')
                running_loss = 0.0
            if counter % SAVE_EVERY == 0:
                torch.save(model.state_dict(), f'{MODEL_NAME}.step_{counter}.pt')

            counter += 1

    return train_losses

train(10)
torch.save(model.state_dict(), f'{MODEL_NAME}.pt')