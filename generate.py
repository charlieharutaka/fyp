import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from models.sodium import Sodium
from models.wavenet import WaveNet
from utils.arpabet import ArpabetEncoding, START, END
from utils.musicxml import parse_musicxml, Note
from hparams import SODIUM_LARGE_HP, WAVENET_HP

encoding = ArpabetEncoding()
all_singers = [
    "female1",
    "female2",
    "female3",
    "female4",
    "female5",
    "female6",
    "female7",
    "female8",
    "female9",
    "male1",
    "male2",
    "male3",
    "male4",
    "male5",
    "male6",
    "male7",
    "male8",
    "male9",
    "male10",
    "male11",
]
singers2idx = dict(zip(all_singers, range(len(all_singers))))
all_techniques = [
    "belt",
    "breathy",
    "fast_forte",
    "fast_piano",
    "forte",
    "inhaled",
    "lip_trill",
    "messa",
    "pp",
    "slow_forte",
    "slow_piano",
    "spoken",
    "straight",
    "trill",
    "trillo",
    "vibrato",
    "vocal_fry"
]
techniques2idx = dict(zip(all_techniques, range(len(all_techniques))))


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


parser = argparse.ArgumentParser(description="Generate audio sample from MusicXML")
parser.add_argument("input", help="Input")
parser.add_argument("-t", "--tacotron-model", help="Tacotron model file", required=True)
parser.add_argument("-w", "--wavenet-model", help="WaveNet model file", required=True)
parser.add_argument("-o", "--output", help="Output", required=True)
parser.add_argument("-S", "--singer", help="Singer", required=True)
parser.add_argument("-T", "--technique", help="Technique", required=True)
parser.add_argument("-s", "--spec-only", action="store_true",
                    help="Don't do wavenet inference (only produce a spectrogram)")
parser.add_argument("-p", "--const-phoneme", help="Constant phoneme if MusicXML contains no lyrics")
parser.add_argument("-g", "--griffinlim", action="store_true", help="Use GL instead of WaveNet")
parser.add_argument("--temperature", help="WaveNet inference temperature", default="0.0")

args = parser.parse_args()

CUDA_IS_AVAILABLE = torch.cuda.is_available()
print(f"Using {'CUDA' if CUDA_IS_AVAILABLE else 'CPU'} | Audio Backend: {torchaudio.get_audio_backend()}")
if CUDA_IS_AVAILABLE:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

score = parse_musicxml(args.input, args.const_phoneme)
tempo = score["P1"]["tempo"]
notes = score["P1"]["notes"]
notes = note_transform(notes)

singer = torch.tensor([singers2idx[args.singer]], dtype=torch.long, device=device)
technique = torch.tensor([techniques2idx[args.technique]], dtype=torch.long, device=device)

lyrics, pitches, durations = zip(*notes)
lyrics = torch.tensor(lyrics, device=device, dtype=torch.long).unsqueeze(1)
pitches = torch.tensor(pitches, device=device).unsqueeze(1)
durations = torch.tensor(durations, device=device).unsqueeze(1)
tempo = torch.tensor([tempo], dtype=torch.float, device=device)

model_tacotron = Sodium(55, 128, 20, 17, **SODIUM_LARGE_HP)
model_tacotron.to(device)
model_tacotron.load_state_dict(torch.load(args.tacotron_model, map_location=device))
model_tacotron.eval()

with torch.no_grad():
    output, output_postnet, weights = model_tacotron.infer(
        singer,
        technique,
        lyrics,
        pitches,
        durations,
        tempo)

output_postnet = output_postnet.squeeze(1)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(
    output_postnet.cpu().transpose(0, 1),
    aspect='auto',
    origin='lower',
    interpolation='nearest',
    cmap='magma')
fig.colorbar(im)
fig.savefig(f'{args.output}.png')

torch.save(output_postnet, f'{args.output}.pt')

if not args.spec_only:
    condition = output_postnet.transpose(0, 1).unsqueeze(0)
    if args.griffinlim:
        inverse_mel_scale = torchaudio.transforms.InverseMelScale(401, 192, f_min=80.0, f_max=8000.0).to(device)
        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=800).to(device)
        spec_condition = inverse_mel_scale(torch.exp(condition))
        audio = griffin_lim(spec_condition)
        audio = audio.squeeze()
    
    else:
        model_wavenet = WaveNet(**WAVENET_HP)
        model_wavenet.to(device)
        model_wavenet.load_state_dict(torch.load(args.wavenet_model, map_location=device))
        model_wavenet.eval()
        encoder = torchaudio.transforms.MuLawEncoding(256)
        decoder = torchaudio.transforms.MuLawDecoding(256)

        hop_len = 400
        GEN_LEN = condition.shape[-1] * hop_len
        TEMP = float(args.temperature)
        generated = [0.0]

        classes = np.arange(256)
        with torch.no_grad():
            model_wavenet.clear_queues(device=device)
            for i in tqdm(range(0, GEN_LEN)):
                inputs = torch.tensor([[[generated[i]]]], device=device)
                local_condition = condition[:,:,i//hop_len].unsqueeze(-1)
                probs = model_wavenet(inputs, local_condition, generate=True).squeeze()
                if TEMP > 0:
                    probs /= TEMP
                    probs = F.softmax(probs, dim=0)
                    output = np.random.choice(classes, p=probs.cpu().numpy(), size=(1,))
                    output = torch.from_numpy(output)
                else:
                    probs = F.softmax(probs, dim=0)
                    output = torch.argmax(probs, dim=-1)
                generated.append(decoder(output).item())
        audio = torch.tensor(generated[1:])
    
    torchaudio.save(f'{args.output}.wav', audio.unsqueeze(0), 16000)