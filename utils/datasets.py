import math
import os
from os import path
from zipfile import ZipFile

import gdown
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio


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
    def __init__(self, root, receptive_field, n_mels=128, n_fft=800):
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
                                  \ \ \ \ \ \ \ \ \ \ \  } Causal
                                   \ \ \ \ \ \ \ \ \ \ \ } Prediction
                                    | | |  target | | | |
                                    |-- segment_size ---|
        The mel-spectrogram data is for the *target* segment.
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
        # The Mel-Spectrogram transform. f_min corresponds to C2 to get rid of very low frequency signals
        self.melspec = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=n_fft, f_min=65.406, pad_mode="constant")
        self.win_length = n_fft
        self.hop_length = self.win_length // 2
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
                    w_length = w.shape[-1]
                    self.original_data.append((w, w_length, w_spec, piece, part, idx))

        # Create chunks of the data: each entry in the spectrogram corresponds to win_length data points.
        # We create segments every hop_length the size of win_length (s.t. there is overlap).
        # Every t'th frame of the spectrogram is centered on time t x hop_length. Hence we must pad the waveform.
        self.length = 0
        self.data = []
        for data in self.original_data:
            num_segments = math.ceil(float(data[1]) / self.hop_length)
            self.length += num_segments
            cur_idx = 0
            cur_ptr = 0
            # # Because torch is useless we have to do reflect padding by ourselves
            # JK we can use constant padding (of 0) since the start/end of the waveforms are close enough to 0
            padded_waveform = F.pad(data[0], (self.hop_length, num_segments * self.hop_length - data[1]), mode="constant")
            # reversed_waveform = data[0].fliplr()
            # padding_left = reversed_waveform[:, -self.hop_length:]
            # padding_right = reversed_waveform[:, :num_segments * self.hop_length - data[1]]
            # padded_waveform = torch.cat((padding_left, data[0], padding_right), dim=1)
            while cur_ptr < padded_waveform.shape[1] - self.hop_length:
                # Calculate where to start the input from for the receptive field
                start_ptr = cur_ptr - self.receptive_field
                input_length = self.receptive_field + self.win_length - 1
                input_padding_length = 0
                if start_ptr < 0:
                    input_padding_length = -start_ptr
                    input_length -= input_padding_length
                    start_ptr = 0
                input = padded_waveform.narrow(1, start_ptr, input_length)
                if input_padding_length > 0:
                    input = F.pad(input, (input_padding_length, 0), mode="constant")

                target = padded_waveform.narrow(1, cur_ptr, self.win_length)
                spec = data[2][:,:,cur_idx]
                self.data.append((input, target, spec, data[3], data[4], data[5]))
                cur_idx += 1
                cur_ptr += self.hop_length

    def __len__(self):
        return self.length
