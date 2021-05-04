from datetime import datetime
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
import torchaudio

import matplotlib.pyplot as plt

from models.tacotron import Tacotron
from utils.datasets import VocalSetDataset, vocal_data_collate_fn
from utils.arpabet import ARPABET, ArpabetEncoding, START, END
from utils.musicxml import Note
from utils.transforms import PowerToDecibelTransform, ScaleToIntervalTransform
from hparams import TACOTRON_HP

CUDA_IS_AVAILABLE = torch.cuda.is_available()
print(f"Using {'CUDA' if CUDA_IS_AVAILABLE else 'CPU'} | Audio Backend: {torchaudio.get_audio_backend()}")
if CUDA_IS_AVAILABLE:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("==========")

BATCH_SIZE=8

print(f"Batch size: {BATCH_SIZE}")

encoding = ArpabetEncoding()

def note_transform(notes):
    notes = [Note(0, 0, [START]), *notes, Note(0, 0, [END])]
    lyrics = []
    pitches = []
    rhythms = []
    for note in notes:
        for phoneme in note.lyric:
            lyrics.append(encoding.encode(phoneme))
            pitches.append(note.pitch)
            rhythms.append(note.duration / len(note.lyric))
    return list(zip(lyrics, pitches, rhythms))

spectrogram_transform = nn.Sequential(PowerToDecibelTransform(torch.max)) #, ScaleToIntervalTransform())
# spectrogram_transform = None
dataset = VocalSetDataset(n_fft=800, n_mels=128, spectrogram_transform=spectrogram_transform, note_transform=note_transform, exclude=["excerpts"])
length_train = int(len(dataset) * 0.9)
length_val = len(dataset) - length_train
dataset_train, dataset_val = random_split(dataset, (length_train, length_val))
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=vocal_data_collate_fn)
loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=vocal_data_collate_fn)

fixed_sample = dataset[0]
fixed_notes = list(zip(*fixed_sample.notes))
fixed_lyrics = torch.tensor(fixed_notes[0]).unsqueeze(0).to(device)
fixed_pitches = torch.tensor(fixed_notes[1]).unsqueeze(0).to(device)
fixed_rhythm = torch.tensor(fixed_notes[2]).unsqueeze(0).to(device)
fixed_notes_len = torch.tensor(fixed_lyrics.shape[1]).unsqueeze(0).to(device)
fixed_mel = fixed_sample.mel.unsqueeze(0).to(device)

def prepare_data(batch):
    notes_lens = batch[6]
    lyrics = []
    pitches = []
    rhythms = []
    for notes in batch[5]:
        lyric, pitch, rhythm = zip(*notes)
        lyrics.append(torch.tensor(lyric))
        pitches.append(torch.tensor(pitch))
        rhythms.append(torch.tensor(rhythm))
    notes_lens = torch.tensor(notes_lens)

    lyrics = nn.utils.rnn.pad_sequence(lyrics, batch_first=True)
    pitches = nn.utils.rnn.pad_sequence(pitches, batch_first=True)
    rhythms = nn.utils.rnn.pad_sequence(rhythms, batch_first=True)

    mels = batch[9]
    mel_lens = batch[10]
    gate_targets = mels.new_zeros((mels.shape[0], mels.shape[1]))
    for batch, mel_len in enumerate(mel_lens):
        gate_targets[batch,mel_len - 1:] = 1.0

    lyrics = lyrics.to(device)
    pitches = pitches.to(device)
    rhythms = rhythms.to(device)
    notes_lens = notes_lens.to(device)
    mels = mels.to(device)
    gate_targets = gate_targets.to(device)

    return lyrics, pitches, rhythms, notes_lens, mels, gate_targets

print(f"Training with {len(loader_train)} batches")

print("==========")

tacotron_hp = { **TACOTRON_HP, "hidden_dim": 128 }

model = Tacotron(num_embeddings=len(encoding), **tacotron_hp)
model.to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
optimizer = torch.optim.Adam(model.parameters())
mel_criterion = nn.MSELoss()
postnet_criterion = nn.MSELoss()
gate_criterion = nn.BCEWithLogitsLoss()
# print(model)

print("==========")

NUM_EPOCHS = 10
PRINT_EVERY = 10
EPOCH_LEN = len(loader_train)

writer = SummaryWriter(f'runs/tacotron/{datetime.now().strftime("%b%d_%H-%M-%S_kaguya")}')
model.train()
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"=== Epoch {epoch} ===")
    losses = []
    model.train()
    for t, batch in enumerate(loader_train, 1):
        optimizer.zero_grad()

        lyrics, pitches, rhythms, notes_lens, mels, gate_targets = prepare_data(batch)

        # notes shape: (batch, sequence)
        # Mels shape: (batch, sequence, hidden_dim)

        output, output_postnet, gate_preds, alignments = model(lyrics, pitches, rhythms, notes_lens, mels)
        mel_loss = mel_criterion(output, mels)
        postnet_loss = postnet_criterion(output_postnet, mels)
        gate_loss = gate_criterion(gate_preds, gate_targets)
        loss = mel_loss + postnet_loss + gate_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # Loss
        losses.append(loss.item())
        global_step = t + (EPOCH_LEN * (epoch - 1))
        writer.add_scalar("Mel Loss", mel_loss.item(), global_step)
        writer.add_scalar("Post-Net Loss", postnet_loss.item(), global_step)
        writer.add_scalar("Gate Loss", gate_loss.item(), global_step)
        writer.add_scalar("Loss", loss.item(), global_step)
        if t % PRINT_EVERY == 0:
            print(f"Loss: {sum(losses) / len(losses)}")
            losses = []
    model.eval()
    with torch.no_grad():
        mel_losses = []
        postnet_losses = []
        gate_losses = []
        losses = []
        for t, batch in enumerate(loader_val, 1):
            lyrics, pitches, rhythms, notes_lens, mels, gate_targets = prepare_data(batch)

            # notes shape: (batch, sequence)
            # Mels shape: (batch, sequence, hidden_dim)

            output, output_postnet, gate_preds, alignments = model(lyrics, pitches, rhythms, notes_lens, mels)
            mel_loss = mel_criterion(output, mels)
            postnet_loss = postnet_criterion(output_postnet, mels)
            gate_loss = gate_criterion(gate_preds, gate_targets)
            loss = mel_loss + postnet_loss + gate_loss

            mel_losses.append(mel_loss.item())
            postnet_losses.append(postnet_loss.item())
            gate_losses.append(gate_loss.item())
            losses.append(loss.item())

        writer.add_scalar("Validation Mel Loss", sum(mel_losses) / len(mel_losses), epoch - 1)
        writer.add_scalar("Validation Post-Net Loss", sum(postnet_losses) / len(postnet_losses), epoch - 1)
        writer.add_scalar("Validation Gate Loss", sum(gate_losses) / len(gate_losses), epoch - 1)
        writer.add_scalar("Validation Loss", sum(losses) / len(losses), epoch - 1)

        print(f"Validation Loss: {sum(losses) / len(losses)}")

        output, output_postnet, gate_preds, alignments = model.infer(fixed_lyrics, fixed_pitches, fixed_rhythm)

        # Alignment illustration
        fig, ax = plt.subplots(figsize=(10,5))
        im = ax.imshow(alignments[0].detach().cpu().transpose(0, 1), aspect='auto', origin='lower', interpolation='nearest')
        fig.colorbar(im)
        writer.add_figure("Alignment", fig, epoch - 1)
        # Spectrogram illustration
        fig, ax = plt.subplots(figsize=(10,5))
        im = ax.imshow(output_postnet[0].detach().cpu().transpose(0, 1), aspect='auto', origin='lower', interpolation='nearest', cmap='magma')
        fig.colorbar(im)
        writer.add_figure("Spectrogram", fig, epoch - 1)
        # Gate value illustration
        fig, ax = plt.subplots(figsize=(10,1))
        im = ax.imshow(torch.sigmoid(gate_preds[0].detach()).cpu().unsqueeze(0), aspect='auto', origin='lower', interpolation='nearest', cmap='RdYlGn')
        fig.colorbar(im)
        writer.add_figure("Gate", fig, epoch - 1)

writer.close()