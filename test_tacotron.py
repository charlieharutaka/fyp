import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

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
dataset = VocalSetDataset(n_fft=800, n_mels=128, spectrogram_transform=spectrogram_transform, note_transform=note_transform)
loader_train = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=vocal_data_collate_fn)

print(len(loader_train))
    
tacotron_hp = {
    "num_embeddings": len(ARPABET),
    "embedding_dim": 64,
    "encoder_p_dropout": 0.5,
    "hidden_dim": 64,
    "attention_dim": 64,
    "attention_rnn_dim": 128,
    "attention_location_n_filters": 8,
    "attention_location_kernel_size": 31,
    "decoder_rnn_dim": 128,
    "p_prenet_dropout": 0.1,
    "p_attention_dropout": 0.1,
    "p_decoder_dropout": 0.1,
    "prenet_dim": 128,
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
gate_criterion = nn.BCELoss()
# print(model)

for batch in loader_train:
    notes_lens = []
    for notes in batch[5]:
        notes_lens.append(len(notes))
    notes = nn.utils.rnn.pad_sequence(batch[5], batch_first=True)
    break

