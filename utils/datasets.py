import torch
from torch.utils.data import Dataset

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
            raise IndexError(f"Index {x} is out of range for dataset size {len(self)}")
        
        waveform = self.waveforms[index]
        relative_index = x - last_cum_segment
        padding_size = self.receptive_field - relative_index
        if padding_size > 0:
            padding = waveform.new_zeros((padding_size,))
            inputs = waveform.narrow(0, 0, relative_index + self.segment_size - 1)
            inputs = torch.cat([padding, inputs], dim=0)
        else:
            inputs = waveform.narrow(0, relative_index - self.receptive_field, self.segment_size + (self.receptive_field - 1))
        target = waveform.narrow(0, relative_index, self.segment_size)

        return inputs, target