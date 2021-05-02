import math
import os
from os import path
import shutil
from zipfile import ZipFile
from collections import Counter, namedtuple
import glob
from dataclasses import dataclass
from typing import List

import gdown
import torch
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset
import torchaudio
from tqdm import tqdm

from .musicxml import parse_musicxml, Note

class SegmentedAudioDataset(Dataset):
    def __init__(self, audio_data, receptive_field, segment_size=32):
        """
        Assumes audio_data is a collection of wave tensors of shape (channels, sequence_length).
        Lazy-loads the data upon calling __getitem__().

        Pads all input tensors by the size of receptive_field.
        Segments the audio into input/target pairs of data:
            |--- receptive_field ---|
                                  |-- segment_size ---|
            | | | | | | | | input | | | | | | | | | | |
                                  \ \ \ \ \ \ \ \ \ \ \  } Causal
                                   \ \ \ \ \ \ \ \ \ \ \ } Prediction
                                    | | |  target | | | |
                                    |-- segment_size ---|
        """

        self.audio_data = audio_data
        self.receptive_field = receptive_field
        self.segment_size = segment_size

        num_segments = []
        cum_segments = []
        waveforms = []
        current_length = 0

        for waveform in audio_data:
            # Make it mono
            waveform = waveform[0]
            # Count the number of possible segments
            current_num_segments = len(waveform) - (segment_size - 1)
            num_segments.append(current_num_segments)
            current_length += current_num_segments
            cum_segments.append(current_length)
            waveforms.append(waveform)

        self.num_segments = num_segments
        self.cum_segments = cum_segments
        self.waveforms = waveforms

    def __len__(self):
        return self.cum_segments[-1]

    def __getitem__(self, x):
        is_bounded = False
        last_cum_segment = 0
        for index, cum_segment in enumerate(self.cum_segments):
            if x < cum_segment:
                is_bounded = True
                break
            last_cum_segment = cum_segment
        if not is_bounded:
            raise IndexError(
                f"Index {x} is out of range for dataset size {len(self)}")

        waveform = self.waveforms[index]
        relative_index = x - last_cum_segment
        padding_size = self.receptive_field - relative_index
        if padding_size > 0:
            padding = waveform.new_zeros((padding_size,))
            inputs = waveform.narrow(
                0, 0, relative_index + self.segment_size - 1)
            inputs = torch.cat([padding, inputs], dim=0)
        else:
            inputs = waveform.narrow(
                0, relative_index - self.receptive_field, self.segment_size + (self.receptive_field - 1))
        target = waveform.narrow(0, relative_index, self.segment_size)

        return inputs, target


class ChoralSingingDataset(Dataset):
    def __init__(self, root, receptive_field, segment_size=None, segment_hop=None, n_mels=64, n_fft=400, spectrogram_transform=None):
        """
        Downloads the choral singing dataset, performs mel-spectrogram analysis on the data.
        Returns samples of segments of the dataset with the corresponding mel-spectrogram and part information.
        The audio is resampled from 44.1kHz to 16kHz for ease of use.
        The audio is also monophonic (1-channel).
        Pads all input tensors by the size of receptive_field.
        Segments the audio into input/target pairs of data:
            |--- receptive_field ---|
                                  |-- segment_size ---|
            | | | | | | | | input | | | | | | | | | | |
                                  \ \ \ \ \ \ \ \ \ \ \   } Causal
                                   \ \ \ \ \ \ \ \ \ \ \  } Prediction
                                    | | |  target | | | |
                                    |-- segment_size ---|
            |     |     |     |     |    |    |    |    | } Spectrogram
            | --- | <- Spectrogram Frame size
        The mel-spectrogram data are of frames from the input sequence used for local conditioning.
        In this case the segment_size is the size of the FFT window.
        """
        # Paths
        self.root = root
        self.subfolder = 'ChoralSingingDataset'
        self.dataset_directory = f'./{self.root}/{self.subfolder}'
        self.dataset_zipfile = f'./{self.root}/{self.subfolder}.zip'
        # Prefix
        self.prefix = "CSD"
        # Audio Extension
        self.ext = "wav"
        # Aliases for the pieces
        self.pieces = {
            "LI": "Locus Iste",
            "ER": "El Rossinyol",
            "ND": "Niño Dios d'Amor Herido"
        }
        # Aliases for the parts
        self.parts = {
            "soprano": "Soprano",
            "alto": "Alto",
            "tenor": "Tenor",
            "bass": "Bass"
        }
        self.original_sample_rate = 44100
        self.target_sample_rate = 16000
        self.resample = torchaudio.transforms.Resample(self.original_sample_rate, self.target_sample_rate)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.melspec = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=n_fft, pad_mode="constant")
        self.win_length = n_fft
        self.hop_length = self.win_length // 2
        self.segment_size = segment_size if segment_size is not None else n_fft
        self.segment_hop = segment_hop if segment_hop is not None else self.segment_size // 2
        self.spectrogram_transform = spectrogram_transform
        # Model receptive field
        self.receptive_field = receptive_field

        # Make the root directory if it doesn't exist
        if not path.exists(self.root):
            print(f"Creating root directory {self.root}")
            os.makedirs(self.root)
        if not path.exists(self.dataset_directory):
            # Assume it's not dowloaded. Download it from google drive
            if not path.exists(self.dataset_zipfile):
                print(f"Downloading dataset to {self.dataset_zipfile}")
                gdown.download(
                    'https://drive.google.com/uc?id=1lp0Ek_ipTm0Goc9-1CPnZpC38bcq10Gx', output=self.dataset_zipfile)
            # The zip file exists but it hasn't been extracted
            print(f"Extracting dataset to {self.dataset_directory}")
            with ZipFile(self.dataset_zipfile) as z:
                z.extractall(self.root)

        # Now the data should be in the right place
        # Load the data
        self.original_data = []
        for piece in self.pieces.keys():
            for part in self.parts.keys():
                for idx in range(4):
                    file_to_load = f"{self.dataset_directory}/{self.prefix}_{piece}_{part}_{idx + 1}.{self.ext}"
                    w, sr = torchaudio.load(file_to_load)
                    assert sr == self.original_sample_rate, "Sample rate mismatch"
                    w = self.resample(w)
                    w_spec = self.melspec(w)
                    if self.spectrogram_transform is not None:
                        w_spec = spectrogram_transform(w_spec)
                    w_length = w.shape[-1]
                    self.original_data.append((w, w_length, w_spec, piece, part, idx))

        self.length = 0
        self.lengths = []
        self.cumulative_lengths = []
        for data in self.original_data:
            num_segments = math.ceil(float(data[1]) / self.segment_hop)
            self.length += num_segments
            self.lengths.append(num_segments)
            self.cumulative_lengths.append(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, x):
        # Find the corresponding data set for the index
        is_bounded = False
        last_cum_len = 0
        for index, cum_len in enumerate(self.cumulative_lengths):
            if x < cum_len:
                is_bounded = True
                break
            last_cum_len = cum_len
        if not is_bounded:
            raise IndexError(
                f"Index {x} is out of range for dataset size {len(self)}")
        offset = x - last_cum_len
        data = self.original_data[index]
        num_segments = self.lengths[index]

        # The padding must be consistent with the mel spectrogram
        padded_waveform = F.pad(data[0], (self.hop_length, num_segments * self.hop_length - data[1]), mode="constant")
        waveform_ptr = self.segment_hop * offset
        start_ptr = waveform_ptr - self.receptive_field
        input_length = self.receptive_field + self.segment_size - 1
        input_padding_length = 0
        if start_ptr < 0:
            input_padding_length = -start_ptr
            input_length -= input_padding_length
            start_ptr = 0
        # Get the input waveform
        input = padded_waveform.narrow(1, start_ptr, input_length)
        if input_padding_length > 0:
            input = F.pad(input, (input_padding_length, 0), mode="constant")
        # Get the target waveform
        target = padded_waveform.narrow(1, waveform_ptr, self.segment_size)
        # Get the spectrograms for the target
        specs = data[2][:,:,offset].squeeze().unsqueeze(-1).expand(-1, input.shape[-1] + 1)
        # The t'th spectrogram is centered on the t * self.hop_length'th waveform data point
        # So input data points from t * self.hop_length - (self.hop_length // 2) to 
        # t * self.hop_length + (self.hop_length // 2) - 1 are associated with frame t
        # TODO: Probably a more efficient way of doing this but CBA
        # spec_indices = [((i + (self.hop_length // 2)) // self.hop_length) for i in range(start_ptr - input_padding_length, waveform_ptr + self.segment_size)]
        # spec_expand_idx_lens = Counter(spec_indices)
        # max_spec_len = data[2].shape[-1]
        # specs = torch.cat([data[2].narrow(2, idx, 1)[0].expand(-1, spec_expand_idx_lens[idx]) if idx >= 0 and idx < max_spec_len else torch.zeros((self.n_mels, spec_expand_idx_lens[idx])) for idx in sorted(spec_expand_idx_lens.keys())], dim=1

        return (input, target, specs,  data[3], data[4], data[5])


@dataclass
class VocalData:
    singer: str
    vocalise: str
    technique: str
    vowel: str
    excerpt: str
    tempo: int
    notes: List[Note]
    wave: torch.Tensor
    mel: torch.Tensor


class VocalSetDataset(Dataset):
    def __init__(self, root='data', n_fft=400, n_mels=64, spectrogram_transform=None, rebuild_cache=False, note_transform=None, exclude=[]):
        self.root = root
        self.subfolder = 'VocalSet'
        self.scores_subfolder = 'VocalSetScores'
        self.dataset_directory = f'./{self.root}/{self.subfolder}'
        self.dataset_scores_directory = f'./{self.root}/{self.scores_subfolder}'
        self.dataset_zipfile = f'./{self.root}/{self.subfolder}.zip'
        self.dataset_scores_zipfile = f'./{self.root}/{self.scores_subfolder}.zip'
        self.cache_dir = f"{self.dataset_directory}/cache"
        # Transforms
        self.resample = torchaudio.transforms.Resample(44100, 16000)
        self.melspec = torchaudio.transforms.MelSpectrogram(16000, n_fft=n_fft, n_mels=n_mels)
        self.spectrogram_transform = spectrogram_transform
        self.note_transform = note_transform
        # Some arpabet things
        self.vowel2arpabet = {
            "a": "AA",
            "e": "EH",
            "i": "IY",
            "o": "OW",
            "u": "UW"
        }
        # Other stuff
        self.exclude = exclude
        # Make the root directory if it doesn't exist
        if not path.exists(self.root):
            print(f"Creating root directory {self.root}")
            os.makedirs(self.root)
        if not path.exists(self.dataset_directory):
            # Assume it's not dowloaded. Download it from google drive
            if not path.exists(self.dataset_zipfile):
                print(f"Downloading dataset to {self.dataset_zipfile}")
                gdown.download(
                    'https://drive.google.com/uc?id=1Ub5nHimDoyuz91azl3iuOidX7zB-ySbf', output=self.dataset_zipfile)
            # The zip file exists but it hasn't been extracted
            print(f"Extracting dataset to {self.dataset_directory}")
            with ZipFile(self.dataset_zipfile) as z:
                z.extractall(self.dataset_directory)
        if not path.exists(self.dataset_scores_directory):
            # Assume it's not dowloaded. Download it from google drive
            if not path.exists(self.dataset_scores_zipfile):
                print(f"Downloading dataset scores to {self.dataset_scores_zipfile}")
                gdown.download(
                    'https://drive.google.com/uc?id=1Hf879juSGqYN3Y0rFmL6cbEwwIlrHa9e', output=self.dataset_scores_zipfile)
            # The zip file exists but it hasn't been extracted
            print(f"Extracting dataset to {self.dataset_scores_directory}")
            with ZipFile(self.dataset_scores_zipfile) as z:
                z.extractall(self.dataset_scores_directory)
        # Build the vocal data
        if not path.exists(self.cache_dir) or rebuild_cache:
            self.data_paths = []
            print("Building dataset cache...")
            if path.exists(self.cache_dir) and rebuild_cache:
                shutil.rmtree(self.cache_dir)
            os.mkdir(self.cache_dir)
            data_paths = glob.glob(f"{self.dataset_directory}/data_by_singer/**/*.wav", recursive=True)
            for exclusion in self.exclude:
                data_paths = filter(lambda path: exclusion not in path, data_paths)
            data_paths = list(data_paths)
            tqdm_iterator = tqdm(data_paths)
            for data_path in tqdm_iterator:
                # Process the wave data
                data_path = path.normpath(data_path)
                wave, sr = torchaudio.load(data_path)
                assert sr == 44100, f"Unexpected sample rate {sr}"
                wave = wave[0]
                wave = self.resample(wave)
                melspec = self.melspec(wave).transpose(0, 1)

                split_path = []
                while data_path != "":
                    data_path, folder = path.split(data_path)
                    split_path.insert(0, folder)
                filename = split_path.pop()
                technique = split_path.pop()
                vocalise = split_path.pop()
                singer = split_path.pop()

                score_path = self.dataset_scores_directory
                if vocalise == "excerpts":
                    excerpt = filename.split("_")[-2].strip()
                    vowel = None
                    score_path += f"/excerpts_{excerpt}.musicxml"
                    score = parse_musicxml(score_path)
                else:
                    filename_split = filename.split("_")
                    vowel = filename_split[-1].split(".")[0]
                    if vowel not in self.vowel2arpabet.keys():
                        tqdm_iterator.set_description(f"Skipping badly-formed file {filename}")
                        continue
                    excerpt = None
                    if vocalise != "long_tones":
                        key = filename_split[2]
                        speed = "fast" if "fast" in technique else "slow"
                        if key != "f":
                            key = "c"
                        vocalise = vocalise + "_" + key + "_" + speed
                    score_path += f"/{vocalise}.musicxml"
                    score = parse_musicxml(score_path, constant_phoneme=self.vowel2arpabet[vowel])
                tempo = score["P1"]["tempo"]
                notes = score["P1"]["notes"]
                assert notes is not None, f"None notes at {filename}"
                data = VocalData(singer, vocalise, technique, vowel, excerpt, tempo, notes, wave, melspec)
                orig_cache_filename = f"{self.cache_dir}/{singer}_{vocalise}_{technique}_{vowel or excerpt}"
                cache_filename = orig_cache_filename + ".pt"
                ctr = 1
                while path.exists(cache_filename):
                    cache_filename = orig_cache_filename + f"_{ctr}.pt"
                    ctr += 1
                torch.save(data, cache_filename)
                self.data_paths.append(cache_filename)
            print(f"Saved {len(self.data_paths)} samples to cache")
        else:
            self.data_paths = glob.glob(f"{self.cache_dir}/*.pt")
            for exclusion in self.exclude:
                self.data_paths = filter(lambda path: exclusion not in path, self.data_paths)
            self.data_paths = list(self.data_paths)
            print(f"Found {len(self.data_paths)} samples in cache")
    
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        path_to_load = self.data_paths[index]
        data = torch.load(path_to_load, map_location='cpu')
        if callable(self.spectrogram_transform):
            data.mel = self.spectrogram_transform(data.mel)
        if callable(self.note_transform):
            data.notes = self.note_transform(data.notes)
        return data
        

def vocal_data_collate_fn(data: List[VocalData]):
    singers = []
    vocalises = []
    techniques = []
    vowels = []
    excerpts = []
    notes = []
    waves = []
    wave_lens = []
    mels = []
    mel_lens = []
    for sample in data:
        singers.append(sample.singer)
        vocalises.append(sample.vocalise)
        techniques.append(sample.technique)
        vowels.append(sample.vowel)
        excerpts.append(sample.excerpt)
        notes.append(sample.notes)
        waves.append(sample.wave)
        wave_lens.append(len(sample.wave))
        mels.append(sample.mel)
        mel_lens.append(len(sample.mel))
    waves = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
    mels = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True)
    return singers, vocalises, techniques, vowels, excerpts, notes, waves, wave_lens, mels, mel_lens
