from datetime import datetime
from pprint import pprint
from itertools import combinations
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
import torchaudio

import matplotlib.pyplot as plt

from models.sodium import Sodium
from utils.datasets import VocalSetDataset, vocal_data_collate_fn
from utils.arpabet import ARPABET, ArpabetEncoding, START, END
from utils.musicxml import Note
from utils.transforms import PowerToDecibelTransform, ScaleToIntervalTransform, DynamicRangeCompression
from utils import round_preserve_sum
from hparams import SODIUM_HP


CUDA_IS_AVAILABLE = torch.cuda.is_available()
print(f"Using {'CUDA' if CUDA_IS_AVAILABLE else 'CPU'} | Audio Backend: {torchaudio.get_audio_backend()}")
if CUDA_IS_AVAILABLE:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("==========")

BATCH_SIZE = 16

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


spectrogram_transform = nn.Sequential(DynamicRangeCompression())  # , ScaleToIntervalTransform())
# spectrogram_transform = None
dataset = VocalSetDataset(
    n_fft=800,
    n_mels=128,
    f_min=80.0,
    f_max=8000.0,
    spectrogram_transform=spectrogram_transform,
    note_transform=note_transform,
    exclude=[])
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
fixed_tempo = torch.tensor([fixed_sample.tempo]).to(device)
fixed_notes_len = torch.tensor(fixed_lyrics.shape[1]).unsqueeze(0).to(torch.int64)
fixed_mel = fixed_sample.mel.unsqueeze(0).to(device)
fixed_mel_len = fixed_mel.shape[1]
fixed_mels_per_beat = fixed_mel_len / fixed_rhythm.sum(dim=1)
fixed_target_durations = round_preserve_sum(fixed_mels_per_beat.unsqueeze(0) * fixed_rhythm)
fixed_computed_tempo = ((1 / fixed_mels_per_beat) / 400) * 16000 * 60
fixed_computed_tempo = fixed_computed_tempo.to(device)

fixed_lyrics = fixed_lyrics.transpose(0, 1)
fixed_pitches = fixed_pitches.transpose(0, 1)
fixed_rhythm = fixed_rhythm.transpose(0, 1)
fixed_mel = fixed_mel.transpose(0, 1)
fixed_target_durations = fixed_target_durations.transpose(0, 1)

print(f"Training with {len(loader_train)} batches")

print("==========")

model = Sodium(num_lyrics=len(encoding), num_pitches=128, **SODIUM_HP)
model.to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
optimizer = torch.optim.Adam(model.parameters())

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
    mel_lens = torch.tensor(mel_lens)

    # approximate which notes belong to which frames:
    mels_per_beat = mel_lens / rhythms.sum(dim=1)
    target_durations = round_preserve_sum(mels_per_beat.unsqueeze(1) * rhythms)
    target_durations = target_durations.to(device)

    lyrics = lyrics.to(device)
    pitches = pitches.to(device)
    rhythms = rhythms.to(device)
    notes_lens = notes_lens.to(torch.int64)
    mels = mels.to(device)
    tempos = torch.tensor(batch[11], dtype=torch.float, device=device)
    computed_tempos = ((1 / mels_per_beat) / 400) * 16000 * 60
    computed_tempos = computed_tempos.to(device)

    return lyrics, pitches, rhythms, notes_lens, mels, tempos, computed_tempos, target_durations


def get_loss(output, output_postnet, pred_durations, mels, target_durations, lambda_dur=1.0):
    # Reconstruction loss
    mel_loss = F.l1_loss(output, mels) + F.mse_loss(output, mels)
    postnet_loss = F.l1_loss(output_postnet, mels) + F.mse_loss(output_postnet, mels)
    recon_loss = mel_loss + postnet_loss
    duration_loss = F.mse_loss(pred_durations, target_durations.to(pred_durations))
    return recon_loss + lambda_dur * duration_loss, mel_loss, postnet_loss, duration_loss


print("==========")

NUM_EPOCHS = 100
PRINT_EVERY = 10
EPOCH_LEN = len(loader_train)

run_name = datetime.now().strftime("%b%d_%H-%M-%S_kaguya")
os.mkdir(run_name)
writer = SummaryWriter(f'runs/sodium/{run_name}')
writer.add_text("Notes", "n_fft=800, n_mels=128, f_min=80.0, f_max=8000.0, clipped on both ends, range clipping 0.1, pitch only no encoder, decoder zoneout")
model.train()

all_mel_losses = []
all_post_mel_losses = []
all_duration_losses = []
all_losses = []

validation_mel_losses = []
validation_post_mel_losses = []
validation_duration_losses = []
validation_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"=== Epoch {epoch} ===")
    losses = []
    model.train()
    for t, batch in enumerate(loader_train, 1):
        optimizer.zero_grad()

        lyrics, pitches, rhythms, notes_lens, mels, tempos, computed_tempos, target_durations = prepare_data(batch)

        lyrics = lyrics.transpose(0, 1)
        pitches = pitches.transpose(0, 1)
        rhythms = rhythms.transpose(0, 1)
        target_durations = target_durations.transpose(0, 1)
        mels = mels.transpose(0, 1)

        output, output_postnet, pred_durations, weights = model(
            lyrics, pitches, rhythms, computed_tempos, target_durations, notes_lens, mels)
        loss, mel_loss, postnet_loss, duration_loss = get_loss(
            output, output_postnet, pred_durations, mels, target_durations, lambda_dur=0.1)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # Loss
        losses.append(loss.item())
        global_step = t + (EPOCH_LEN * (epoch - 1))
        writer.add_scalar("Mel Loss", mel_loss.item(), global_step)
        writer.add_scalar("Post-Net Loss", postnet_loss.item(), global_step)
        writer.add_scalar("Duration Loss", duration_loss.item(), global_step)
        writer.add_scalar("Loss", loss.item(), global_step)

        all_mel_losses.append(mel_loss.item())
        all_post_mel_losses.append(postnet_loss.item())
        all_duration_losses.append(duration_loss.item())
        all_losses.append(loss.item())

        if t % PRINT_EVERY == 0:
            print(f"Loss: {sum(losses) / len(losses)}")
            losses = []
    model.eval()
    with torch.no_grad():
        mel_losses = []
        postnet_losses = []
        duration_losses = []
        losses = []
        for t, batch in enumerate(loader_val, 1):
            lyrics, pitches, rhythms, notes_lens, mels, tempos, computed_tempos, target_durations = prepare_data(batch)

            lyrics = lyrics.transpose(0, 1)
            pitches = pitches.transpose(0, 1)
            rhythms = rhythms.transpose(0, 1)
            target_durations = target_durations.transpose(0, 1)
            mels = mels.transpose(0, 1)

            output, output_postnet, pred_durations, weights = model(
                lyrics, pitches, rhythms, computed_tempos, target_durations, notes_lens, mels)
            loss, mel_loss, postnet_loss, duration_loss = get_loss(
                output, output_postnet, pred_durations, mels, target_durations, lambda_dur=0.1)

            mel_losses.append(mel_loss.item())
            postnet_losses.append(postnet_loss.item())
            duration_losses.append(duration_loss.item())
            losses.append(loss.item())

        writer.add_scalar("Validation Mel Loss", sum(mel_losses) / len(mel_losses), epoch - 1)
        writer.add_scalar("Validation Post-Net Loss", sum(postnet_losses) / len(postnet_losses), epoch - 1)
        writer.add_scalar("Validation Duration Loss", sum(duration_losses) / len(duration_losses), epoch - 1)
        writer.add_scalar("Validation Loss", sum(losses) / len(losses), epoch - 1)

        validation_mel_losses.append(sum(mel_losses) / len(mel_losses))
        validation_post_mel_losses.append(sum(postnet_losses) / len(postnet_losses))
        validation_duration_losses.append(sum(duration_losses) / len(duration_losses))
        validation_losses.append(sum(losses) / len(losses))

        print(f"Validation Loss: {sum(losses) / len(losses)}")

        # Forced illustrations
        output, output_postnet, pred_durations, weights = model(
            fixed_lyrics, fixed_pitches, fixed_rhythm, fixed_computed_tempo, fixed_target_durations, fixed_notes_len, fixed_mel)

        # Duration illustration
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(weights[:,0,:].cpu(),
                       aspect='auto', origin='lower', interpolation='nearest')
        fig.colorbar(im)
        writer.add_figure("Forced Duration Weights", fig, epoch - 1)
        fig.savefig(f'{run_name}/forced.duration.{epoch}.png')
        
        # Spectrogram illustration
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(
            output_postnet[:,0].cpu().transpose(
                0,
                1),
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='magma')
        fig.colorbar(im)
        writer.add_figure("Forced Spectrogram", fig, epoch - 1)
        fig.savefig(f'{run_name}/forced.spectrogram.{epoch}.png')

        # Inference illustrations
        output, output_postnet, weights = model.infer(
            fixed_lyrics, fixed_pitches, fixed_rhythm, fixed_computed_tempo)

        # Duration illustration
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(weights[:,0,:].cpu(),
                       aspect='auto', origin='lower', interpolation='nearest')
        fig.colorbar(im)
        writer.add_figure("Inferred Duration Weights", fig, epoch - 1)
        fig.savefig(f'{run_name}/infer.duration.{epoch}.png')
        
        # Spectrogram illustration
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(
            output_postnet[:, 0].cpu().transpose(
                0,
                1),
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='magma')
        fig.colorbar(im)
        writer.add_figure("Inferred Spectrogram", fig, epoch - 1)
        fig.savefig(f'{run_name}/infer.spectrogram.{epoch}.png')

    torch.save(model.state_dict(), f"{run_name}/sodium.{epoch}.pt")

    torch.save(all_mel_losses, f"{run_name}/all_mel_losses.pt")
    torch.save(all_post_mel_losses, f"{run_name}/all_post_mel_losses.pt")
    torch.save(all_duration_losses, f"{run_name}/all_duration_losses.pt")
    torch.save(all_losses, f"{run_name}/all_losses.pt")

    torch.save(validation_mel_losses, f"{run_name}/validation_mel_losses.pt")
    torch.save(validation_post_mel_losses, f"{run_name}/validation_post_mel_losses.pt")
    torch.save(validation_duration_losses, f"{run_name}/validation_duration_losses.pt")
    torch.save(validation_losses, f"{run_name}/validation_losses.pt")

writer.close()
