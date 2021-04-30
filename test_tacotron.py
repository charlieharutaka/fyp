import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

import matplotlib.pyplot as plt

from models.tacotron import Tacotron
from utils.datasets import VocalSetDataset, vocal_data_collate_fn
from utils.arpabet import ARPABET, ArpabetEncoding
from utils.transforms import PowerToDecibelTransform, ScaleToIntervalTransform

CUDA_IS_AVAILABLE = torch.cuda.is_available()
print(f"Using {'CUDA' if CUDA_IS_AVAILABLE else 'CPU'} | Audio Backend: {torchaudio.get_audio_backend()}")
if CUDA_IS_AVAILABLE:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

encoding = ArpabetEncoding()

note_transform = lambda notes: torch.cat([torch.tensor([encoding.encode(phoneme) for phoneme in note.lyric]) for note in notes])
spectrogram_transform = nn.Sequential(PowerToDecibelTransform(torch.max), ScaleToIntervalTransform())
dataset = VocalSetDataset(n_fft=800, n_mels=128, spectrogram_transform=spectrogram_transform, note_transform=note_transform, exclude=["excerpts"])
loader_train = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=vocal_data_collate_fn)

print(f"Training with {len(loader_train)} batches")

print("==========")

tacotron_hp = {
    "num_embeddings": len(ARPABET),
    "embedding_dim": 64,
    "encoder_p_dropout": 0.5,
    "hidden_dim": 128, # Mels
    "attention_dim": 128,
    "attention_rnn_dim": 512,
    "attention_location_n_filters": 16,
    "attention_location_kernel_size": 31,
    "decoder_rnn_dim": 512,
    "p_prenet_dropout": 0.1,
    "p_attention_dropout": 0.1,
    "p_decoder_dropout": 0.1,
    "prenet_dim": 512,
    "max_decoder_steps": 1000,
    "stopping_threshold": 0.5,
    "postnet_n_convolutions": 5,
    "postnet_embedding_dim": 512,
    "postnet_kernel_size": 5,
    "postnet_p_dropout": 0.5
}

model = Tacotron(**tacotron_hp)
model.to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
optimizer = torch.optim.Adam(model.parameters())
mel_criterion = nn.MSELoss()
gate_criterion = nn.BCEWithLogitsLoss()
# print(model)

print("==========")

for batch in loader_train:
    optimizer.zero_grad()

    notes_lens = []
    for notes in batch[5]:
        notes_lens.append(len(notes))
    notes_lens = torch.tensor(notes_lens)
    notes = nn.utils.rnn.pad_sequence(batch[5], batch_first=True)
    mels = batch[8]
    mel_lens = batch[9]
    gate_targets = mels.new_zeros((mels.shape[0], mels.shape[1]))
    for batch, mel_len in enumerate(mel_lens):
        gate_targets[batch,mel_len - 1:] = 1.0
    # notes shape: (batch, sequence)
    # Mels shape: (batch, sequence, hidden_dim)
    output, output_postnet, gate_preds, alignments = model(notes, notes_lens, mels)
    print(mels.shape, gate_targets.shape)
    print(output.shape, output_postnet.shape, gate_preds.shape, alignments.shape)
    # fig = plt.imshow(alignments[0].detach().cpu().transpose(0, 1), aspect='auto', origin='lower')
    recon_loss = mel_criterion(output_postnet, mels)
    gate_loss = gate_criterion(gate_preds, gate_targets)
    loss = recon_loss + gate_loss
    print(loss.item())
    loss.backward()
    optimizer.step()
    break

