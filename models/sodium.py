import math
from typing import Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from .tacotron import get_mask_from_lengths
from .layers import ZoneOutLSTM
from .utils import softclamp

"""
Non-attentive Tacotron aka Sodium (because Na get it ha ha)

Encoder:
- Embeddings:
    - Lyrics Embedding
    - Pitch Embedding
    - Duration Embedding
- Pre-Net:
    - 3 x Conv with kernel size 5
- Either:
    - Transformer:
        - Positional Embedding
        - Transformer Encoder
    - BiLSTM
- Gaussian Upsampling:
    - BiLSTM + Projection
    - Follow the gaussian upsampling algorithm
- Positional Encoding:
    - Given the duration embedding, produce positional encodings
Decoder:
- LSTM + Projection
- Pre-Net
Post-Net:
- 5 x Conv with kernel size 5


"""


""" ----- Utility Modules ----- """


class TransformerPositionalEncoding(nn.Module):
    """
    Positional encoder layer for Transformer networks.
    Given an input X, add positional information of the form:

        X[pos, 2i]      += sin(pos / 10000 ^ (2i / dim))
        X[pos, 2i + 1]  += cos(pos / 10000 ^ (2i / dim))

    The positional encoding is saved as a constant buffer to save computation cycles.
    It is initalized with a fixed number of positions but will lazily expand if it encounters longer sequences.
    """

    def __init__(self, dim: int, denom: float = 10000.0, max_length: int = 1000):
        super(TransformerPositionalEncoding, self).__init__()
        self.dim = dim
        self.denom = denom
        encoding = torch.empty((max_length, dim))
        pos = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(denom) / dim))

        encoding[:, 0::2] = torch.sin(pos * div)
        encoding[:, 1::2] = torch.cos(pos * div)
        encoding = encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('encoding', encoding)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: (sequence, batch, dim)
        Returns:
            x: (sequence, batch, dim)
        """
        x = x + self.encoding[:x.shape[0], :]
        return x


class PositionalEmbedding(nn.Module):
    """
    Positional embedding layer for Non-Attentive Tacotron.

    "   The positional embedding tracks the index of each upsampled frame within
        each token; if the duration values are [2, 1, 3], the indices for the
        positional embedding would be [1, 2, 1, 1, 2, 3]. "

    Given duration values D, return positional information as described above.
    """

    def __init__(self, dim: int, denom: float = 10000.0, max_length: int = 1000):
        super(PositionalEmbedding, self).__init__()
        self.dim = dim
        self.denom = denom
        encoding = torch.empty((max_length, dim))
        pos = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(denom) / dim))

        encoding[:, 0::2] = torch.sin(pos * div)
        encoding[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('encoding', encoding)

    def forward(self, durations: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            durations: (sequence_in, batch)
        Returns:
            embedding: (sequence_out, batch, dim)
        """
        embedding = []
        durations = durations.transpose(0, 1)  # easier to do batch-first
        for batch in range(durations.shape[0]):
            seq_embedding = []
            for duration in durations[batch]:
                seq_embedding.append(self.encoding[:duration, :])
            seq_embedding = torch.cat(seq_embedding, dim=0)
            embedding.append(seq_embedding)
        embedding = nn.utils.rnn.pad_sequence(embedding)
        return embedding


class GaussianUpsampling(nn.Module):
    """
    Gaussian upsampling as described in the Non-Attentive Tacotron paper.

    Given an input sequence H, duration values D and range parameters S, we can
    upsample it by computing:

        C[i] = (D[i] / 2) + sum(D[:i])
        W[t, i] = N(t; C[i], S[i] ^ 2) / sum(N(t; C[*], S[*] ^ 2))
        U[t] = sum(W[t, *] * H[*])

    In essence, for each t we are assuming a Gaussian distribution with mean t
    and std. dev. s. This mixture distribution determines the mixture of frames
    in H that compose U.
    """

    def __init__(self):
        super(GaussianUpsampling, self).__init__()

    def forward(self, inputs: torch.Tensor, durations: torch.LongTensor,
                ranges: torch.Tensor) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Note:
            Padding is assumed to be 0 and hence does not contribute to cumsum
        Args:
            inputs: (sequence_in, batch, dim)
            durations: (sequence_in, batch)
            ranges: (sequence_in, batch)
        Returns:
            upscaled: (sequence_out, batch)
            sequence_lengths: (batch,)
            weights: (sequence_in, batch, sequence_out)
        """
        # It's easier to do batch first for this stuff
        inputs = inputs.transpose(0, 1)
        durations = durations.transpose(0, 1)
        ranges = ranges.transpose(0, 1)

        # Get the sizes
        batch, sequence_in, dim = inputs.shape

        # Sum in the sequence dimension & subtract a half of the duration
        cum_durations = torch.cumsum(durations, dim=1)
        sequence_out = torch.max(cum_durations).item()
        sequence_lengths = cum_durations[:, -1].squeeze()
        cum_durations, durations = cum_durations.to(inputs), durations.to(inputs)
        cum_durations = cum_durations - (durations / 2)

        # Construct the normal distribution to compute log-probabilities from
        normal = D.Normal(cum_durations.unsqueeze(-1), ranges.unsqueeze(-1))
        normal = normal.expand((batch, sequence_in, sequence_out))

        # Enumerate the output indices: (sequence_out,)
        ts = torch.arange(0, sequence_out).to(inputs)

        # compute the log-probabilities over all i, t
        probs = normal.log_prob(ts)  # (batch, sequence_in, sequence_out)

        # Sum the probabilities along the input dimension
        cum_probs = probs.logsumexp(dim=1).unsqueeze(1)
        weights = probs - cum_probs  # (batch, sequence_in, sequence_out)
        weights = weights.exp()

        # Upscale the input using the weights matrix
        upscaled = torch.bmm(weights.transpose(1, 2), inputs)

        # Back to sequence-first
        upscaled = upscaled.transpose(0, 1)
        weights = weights.transpose(0, 1)

        return upscaled, sequence_lengths, weights


""" ----- Encoder Modules ----- """


class SodiumEncoderPrenet(nn.Module):
    def __init__(
            self,
            num_convolutions: int = 3,
            input_features: int = 256,
            output_features: int = 256,
            kernel_size: int = 5,
            activation: nn.Module = nn.Identity()):
        """
        Simple pre-net that applies sequential convolutions with batch norm
        """
        super(SodiumEncoderPrenet, self).__init__()
        padding = (kernel_size - 1) // 2
        self.num_convolutions = num_convolutions
        self.convolutions = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_convolutions):
            self.convolutions.append(
                nn.Conv1d(
                    input_features if i == 0 else output_features,
                    output_features,
                    kernel_size,
                    padding=padding))
            self.batch_norms.append(nn.BatchNorm1d(output_features))
        self.activation = activation

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: (batch, input_features, sequence)
        Returns:
            x: (batch, output_features, sequence)
        """
        for i in range(self.num_convolutions):
            x = self.convolutions[i](x)
            x = self.batch_norms[i](x)
            x = self.activation(x)
        return x


class SodiumEncoder(nn.Module):
    def __init__(
            self,
            num_lyrics: int,
            num_pitches: int,
            embedding_lyric_dim: int = 256,
            embedding_pitch_dim: int = 256,
            embedding_dim: int = 256,
            prenet_num_convolutions: int = 3,
            prenet_kernel_size: int = 5,
            prenet_activation: nn.Module = nn.Identity(),
            p_dropout: float = 0.1,
            use_transformer: bool = True,
            transformer_nlayers: int = 8,
            transformer_nhead: int = 4,
            transformer_ff_dim: int = 1024,
            transformer_activation: str = "relu",
            lstm_num_layers: int = 1,
            lstm_zoneout: float = 0.1):
        super(SodiumEncoder, self).__init__()
        self.embedding_lyrics = nn.Embedding(num_lyrics, embedding_lyric_dim)
        self.embedding_pitches = nn.Embedding(num_pitches, embedding_pitch_dim)

        prenet_input_dim = embedding_lyric_dim + embedding_pitch_dim
        self.prenet = SodiumEncoderPrenet(
            num_convolutions=prenet_num_convolutions,
            input_features=prenet_input_dim,
            output_features=embedding_dim,
            kernel_size=prenet_kernel_size,
            activation=prenet_activation)

        self.use_transformer = use_transformer
        if use_transformer:
            transformer_layer = nn.TransformerEncoderLayer(
                embedding_dim, transformer_nhead, transformer_ff_dim, p_dropout, transformer_activation)
            self.pos_enc = TransformerPositionalEncoding(embedding_dim)
            self.encoder = nn.TransformerEncoder(transformer_layer, transformer_nlayers)
        else:
            self.encoder = ZoneOutLSTM(
                input_size=embedding_dim,
                hidden_size=(
                    embedding_dim // 2),
                bidirectional=True,
                num_layers=lstm_num_layers,
                zoneout=lstm_zoneout)

    def forward(
            self,
            lyrics: torch.LongTensor,
            pitches: torch.LongTensor,
            input_lengths: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            lyrics: (sequence, batch)
            pitches: (sequence, batch)
            input_lengths: (batch)
        Returns:
            output (sequence, batch, embedding_dim)
        """
        lyrics = self.embedding_lyrics(lyrics)
        pitches = self.embedding_pitches(pitches)
        # concat and pass through prenet
        prenet_input = torch.cat([lyrics, pitches], dim=-1)
        prenet_input = prenet_input.permute(1, 2, 0)
        prenet_output = self.prenet(prenet_input)
        prenet_output = prenet_output.permute(2, 0, 1)

        if self.use_transformer:
            # we use the positional encoding
            encoder_in = self.pos_enc(prenet_output)
            input_mask = ~get_mask_from_lengths(input_lengths)
            input_mask = input_mask.to(encoder_in.device)
            output = self.encoder(encoder_in, src_key_padding_mask=input_mask)
        else:
            output = nn.utils.rnn.pack_padded_sequence(prenet_output, input_lengths)
            self.encoder.flatten_parameters()
            output, _ = self.encoder(output)
            output, _ = nn.utils.rnn.pad_packed_sequence(output)

        return output

    def infer(
            self,
            lyrics: torch.LongTensor,
            pitches: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            lyrics: (sequence, batch)
            pitches: (sequence, batch)
        Returns:
            output (sequence, batch, embedding_dim)
        """
        lyrics = self.embedding_lyrics(lyrics)
        pitches = self.embedding_pitches(pitches)
        # concat and pass through prenet
        prenet_input = torch.cat([lyrics, pitches], dim=-1)
        prenet_input = prenet_input.permute(1, 2, 0)
        prenet_output = self.prenet(prenet_input)
        prenet_output = prenet_output.permute(2, 0, 1)

        if self.use_transformer:
            # we use the positional encoding
            encoder_in = self.pos_enc(prenet_output)
            output = self.encoder(encoder_in)
        else:
            output, _ = self.encoder(prenet_output)

        return output


class SodiumDurationPredictor(nn.Module):
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 256, n_layers: int = 1, bias: bool = False):
        super(SodiumDurationPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim + 1,
            hidden_size=(
                hidden_dim // 2),
            bidirectional=True,
            num_layers=n_layers,
            bias=bias)
        self.projection = nn.Linear(hidden_dim, 1, bias=bias)

    def forward(
            self,
            encoder_output: torch.FloatTensor,
            durations: torch.FloatTensor,
            tempo: torch.FloatTensor,
            input_lengths: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward function for training purposes.
        Args:
            encoder_output: (sequence, batch, embedding_dim)
            durations: (sequence, batch)
            tempo: (batch)
            input_lengths: (batch)
        Returns:
            pred_durations: (sequence, batch)
        """
        # we compute the 'true' duration by dividing the duration tensor by the tempo tensor (beats / bpm -> duration)
        # TODO: think about this some more
        durations = (durations / tempo) * 60.0
        durations = durations.unsqueeze(-1)
        # TODO: is it really necessary to have an entire LSTM for this??
        lstm_in = torch.cat([durations, encoder_output], dim=-1)
        lstm_in = nn.utils.rnn.pack_padded_sequence(lstm_in, input_lengths)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out)
        # Projection
        pred_durations = F.relu(self.projection(lstm_out))
        return pred_durations.squeeze(2)

    def infer(
            self,
            encoder_output: torch.FloatTensor,
            durations: torch.FloatTensor,
            tempo: torch.FloatTensor) -> torch.LongTensor:
        """
        Args:
            encoder_output: (sequence, batch, embedding_dim)
            durations: (sequence, batch)
            tempo: (batch)
        Returns:
            pred_durations: (sequence, batch)
        """
        durations = (durations / tempo) * 60.0
        durations = durations.unsqueeze(-1)

        lstm_in = torch.cat([durations, encoder_output], dim=-1)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(lstm_in)

        pred_durations = F.relu(self.projection(lstm_out))
        return pred_durations.squeeze(2)


class SodiumRangePredictor(nn.Module):
    """
    Module that predicts the range parameter (stdev) for the Gaussian upsampling.
    If clip is a float greater than 0, the output ranges are clipped between 0 and
    clip times the duration value of that range.
    """
    def __init__(
            self,
            embedding_dim: int = 256,
            hidden_dim: int = 256,
            n_layers: int = 1,
            bias: bool = True,
            clip: float = 0.0):
        super(SodiumRangePredictor, self).__init__()
        self.clip = clip
        self.lstm = nn.LSTM(
            input_size=embedding_dim + 1,
            hidden_size=(
                hidden_dim // 2),
            bidirectional=True,
            num_layers=n_layers,
            bias=bias)
        self.projection = nn.Linear(hidden_dim, 1, bias=bias)

    def forward(
            self,
            encoder_output: torch.FloatTensor,
            durations_output: torch.LongTensor,
            input_lengths: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            durations_output: (sequence, batch)
            encoder_output: (sequence, batch, embedding_dim)
            input_lengths: (batch)
        Returns:
            pred_ranges: (sequence, batch)
        """
        durations_output = durations_output.unsqueeze(-1).to(torch.float)
        lstm_in = torch.cat([durations_output, encoder_output], dim=-1)
        lstm_in = nn.utils.rnn.pack_padded_sequence(lstm_in, input_lengths)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out)

        pred_ranges = self.projection(lstm_out)
        # if self.clip > 0.0:
        #     pred_ranges = softclamp(pred_ranges, self.clip * durations_output)
        #     print(pred_ranges)
        # else:
        if self.clip > 0.0:
            pred_ranges = torch.minimum(pred_ranges, self.clip * durations_output)
        pred_ranges = F.softplus(pred_ranges)
        return pred_ranges

    def infer(self, encoder_output: torch.FloatTensor, durations_output: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            encoder_output: (sequence, batch, embedding_dim)
            durations_output: (sequence, batch)
        Returns:
            pred_ranges: (sequence, batch)
        """
        durations_output = durations_output.unsqueeze(-1).to(torch.float)
        lstm_in = torch.cat([durations_output, encoder_output], dim=-1)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(lstm_in)
        pred_ranges = F.softplus(self.projection(lstm_out))
        return pred_ranges


class SodiumUpsampler(nn.Module):
    """
    Module that takes the encoder output, predicts duration & range values,
    then upscales it and adds positional embeddings.
    """

    def __init__(
            self,
            embedding_dim: int = 256,
            duration_hidden_dim: int = 256,
            duration_n_layers: int = 1,
            duration_bias: bool = False,
            range_hidden_dim: int = 256,
            range_n_layers: int = 1,
            range_bias: bool = True,
            range_clip: float = 0.0,
            pos_embedding_dim: int = 32,
            pos_embedding_denom: float = 10000.0,
            pos_embedding_max_len: int = 1000):
        super(SodiumUpsampler, self).__init__()
        self.duration_predictor = SodiumDurationPredictor(
            embedding_dim=embedding_dim,
            hidden_dim=duration_hidden_dim,
            n_layers=duration_n_layers,
            bias=duration_bias)
        self.range_predictor = SodiumRangePredictor(
            embedding_dim=embedding_dim,
            hidden_dim=range_hidden_dim,
            n_layers=range_n_layers,
            bias=range_bias,
            clip=range_clip)
        self.pos_embedding = PositionalEmbedding(
            dim=pos_embedding_dim,
            denom=pos_embedding_denom,
            max_length=pos_embedding_max_len)
        self.upsampler = GaussianUpsampling()

    def forward(
        self,
        encoder_output: torch.FloatTensor,
        durations: torch.FloatTensor,
        tempo: torch.FloatTensor,
        target_durations: torch.LongTensor,
        input_lengths: torch.LongTensor) -> Tuple[torch.FloatTensor,
                                                  torch.FloatTensor,
                                                  torch.LongTensor,
                                                  torch.FloatTensor]:
        """
        Args:
            encoder_output: (sequence_in, batch, embedding_dim)
            durations: (sequence_in, batch)
            tempo: (batch)
            target_durations: (sequence_in, batch)
            input_lengths: (batch)
        Returns:
            pred_durations: (sequence_in, batch)
            upsampled: (sequence_out, batch, embedding_dim + pos_embedding_dim)
            upsampled_lengths: (batch)
            weights: (sequence_in, batch, sequence_out)
        """
        pred_durations = self.duration_predictor(encoder_output, durations, tempo, input_lengths)
        pred_ranges = self.range_predictor(encoder_output, target_durations, input_lengths).squeeze(2)
        upsampled, upsampled_lengths, weights = self.upsampler(encoder_output, target_durations, pred_ranges)
        pe = self.pos_embedding(target_durations)
        upsampled = torch.cat([upsampled, pe], dim=-1)

        return pred_durations, upsampled, upsampled_lengths, weights

    def infer(
        self,
        encoder_output: torch.FloatTensor,
        durations: torch.FloatTensor,
        tempo: torch.FloatTensor) -> Tuple[torch.FloatTensor,
                                           torch.LongTensor,
                                           torch.FloatTensor]:
        """
        Args:
            encoder_output: (sequence_in, batch, embedding_dim)
            durations: (sequence_in, batch)
            tempo: (batch)
        Returns:
            upsampled: (sequence_out, batch, embedding_dim + pos_embedding_dim)
            upsampled_lengths: (batch)
            weights: (sequence_in, batch, sequence_out)
        """
        pred_durations = self.duration_predictor.infer(encoder_output, durations, tempo)
        pred_durations = pred_durations.to(torch.long)
        pred_ranges = self.range_predictor.infer(encoder_output, pred_durations).squeeze(2)
        upsampled, upsampled_lengths, weights = self.upsampler(encoder_output, pred_durations, pred_ranges)
        pe = self.pos_embedding(pred_durations)
        upsampled = torch.cat([upsampled, pe], dim=-1)

        return upsampled, upsampled_lengths, weights


""" ---- Decoder Modules ---- """


class SodiumDecoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 256,
            prenet_n_layers: int = 2,
            prenet_dim: int = 256,
            prenet_activation: nn.Module = nn.ReLU(),
            prenet_p_dropout: float = 0.5,
            decoder_dim: int = 1024,
            decoder_n_layers: int = 1,
            output_dim: int = 128):
        super(SodiumDecoder, self).__init__()
        self.decoder_dim = decoder_dim
        self.decoder_n_layers = decoder_n_layers
        self.output_dim = output_dim
        # Pre-Net Projection
        pre_net = []
        for i in range(prenet_n_layers):
            pre_net.append(nn.Sequential(
                nn.Linear(output_dim if i == 0 else prenet_dim, prenet_dim),
                prenet_activation,
                nn.Dropout(prenet_p_dropout)))
        self.pre_net = nn.Sequential(*pre_net)
        # Decoder
        self.decoder = nn.ModuleList([
            nn.LSTMCell(input_size=prenet_dim + embedding_dim if i == 0 else decoder_dim,
                        hidden_size=decoder_dim)
            for i in range(decoder_n_layers)])
        self.projection = nn.Linear(decoder_dim + embedding_dim, output_dim)

    def init_go_frame(self, encoder_out: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns an all-zero frame <GO>
        Args:
            encoder_out (sequence, batch, embedding_dim)
        Returns:
            go_frame (batch, output_dim)
        """
        _, batch, _ = encoder_out.shape
        return encoder_out.new_zeros((batch, self.output_dim), requires_grad=True)

    def init_rnn_states(self, encoder_out: torch.FloatTensor) -> None:
        """
        Initializes the LSTM hidden, cell states for each layer in the decoder
        Args:
            encoder_out (sequence, batch, embedding_dim)
        Returns:
            None
        """
        _, batch, _ = encoder_out.shape
        self.hidden_states = [
            encoder_out.new_zeros(
                (batch, self.decoder_dim), requires_grad=True) for _ in range(
                self.decoder_n_layers)]
        self.cell_states = [
            encoder_out.new_zeros(
                (batch, self.decoder_dim), requires_grad=True) for _ in range(
                self.decoder_n_layers)]

    def forward(self, encoder_out: torch.FloatTensor, mels: torch.FloatTensor) -> torch.FloatTensor:
        """
        Teacher-forced training.
        Args:
            encoder_out: (sequence, batch, embedding_dim)
            mels: (sequence, batch, output_dim)
        Returns:
            output: (sequence, batch, output_dim)
        """
        go_frame = self.init_go_frame(encoder_out)
        decoder_inputs = torch.cat([go_frame.unsqueeze(0), mels], dim=0)
        decoder_inputs = self.pre_net(decoder_inputs)

        self.init_rnn_states(encoder_out)
        outputs = []
        for frame in range(encoder_out.shape[0]):
            output = torch.cat((decoder_inputs[frame], encoder_out[frame]), dim=-1)
            # output has size (batch, prenet_dim + embedding_dim)
            for layer in range(self.decoder_n_layers):
                self.hidden_states[layer], self.cell_states[layer] =\
                    self.decoder[layer](output, (self.hidden_states[layer], self.cell_states[layer]))
                output = self.hidden_states[layer]
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        output = torch.cat((outputs, encoder_out), dim=-1)
        output = self.projection(output)

        return output

    def infer(self, encoder_out: torch.FloatTensor) -> torch.FloatTensor:
        """
        Inference.
        Args:
            encoder_out: (sequence, batch, embedding_dim)
        Returns:
            output: (sequence, batch, output_dim)
        """
        output = self.init_go_frame(encoder_out)
        self.init_rnn_states(encoder_out)
        outputs = []
        for frame in range(encoder_out.shape[0]):
            output = self.pre_net(output)
            output = torch.cat([output, encoder_out[frame]], dim=-1)
            for layer in range(self.decoder_n_layers):
                self.hidden_states[layer], self.cell_states[layer] =\
                    self.decoder[layer](output, (self.hidden_states[layer], self.cell_states[layer]))
                output = self.hidden_states[layer]
            output = torch.cat([output, encoder_out[frame]], dim=-1)
            output = self.projection(output)
            outputs.append(output)
        if len(outputs) > 0:
            return torch.stack(outputs, dim=0)
        else:
            warnings.warn("No mels produced", UserWarning)
            return torch.empty((0, encoder_out.shape[1], self.output_dim))


class SodiumPostnet(nn.Module):
    def __init__(
            self,
            num_convolutions: int = 5,
            features: int = 128,
            hidden_features: int = 512,
            kernel_size: int = 5,
            activation: nn.Module = nn.Tanh()):
        """
        Simple post-net that applies sequential convolutions with batch norm.
        Predicts a residual to add to the mel-spectrogram.
        """
        super(SodiumPostnet, self).__init__()
        padding = (kernel_size - 1) // 2
        self.num_convolutions = num_convolutions
        self.convolutions = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_convolutions):
            self.convolutions.append(
                nn.Conv1d(
                    features if i == 0 else hidden_features,
                    features if i == num_convolutions - 1 else hidden_features,
                    kernel_size,
                    padding=padding))
            if i != num_convolutions - 1:
                self.batch_norms.append(nn.BatchNorm1d(hidden_features))
        self.activation = activation

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: (batch, features, sequence)
        Returns:
            x: (batch, features, sequence)
        """
        for i in range(self.num_convolutions - 1):
            x = self.convolutions[i](x)
            x = self.batch_norms[i](x)
            x = self.activation(x)
        x = self.convolutions[-1](x)
        return x


""" ---- The saltiest NN known to mankind ---- """


class Sodium(nn.Module):
    def __init__(
            self,
            num_lyrics: int,
            num_pitches: int,
            embedding_lyric_dim: int = 256,
            embedding_pitch_dim: int = 256,
            embedding_dim: int = 256,
            encoder_prenet_num_convolutions: int = 3,
            encoder_prenet_kernel_size: int = 5,
            encoder_prenet_activation: nn.Module = nn.Identity(),
            encoder_p_dropout: float = 0.1,
            encoder_use_transformer: bool = True,
            encoder_transformer_nlayers: int = 8,
            encoder_transformer_nhead: int = 4,
            encoder_transformer_ff_dim: int = 1024,
            encoder_transformer_activation: str = "relu",
            encoder_lstm_num_layers: int = 1,
            encoder_lstm_zoneout: float = 0.1,
            duration_hidden_dim: int = 256,
            duration_n_layers: int = 1,
            duration_bias: bool = False,
            range_hidden_dim: int = 256,
            range_n_layers: int = 1,
            range_bias: bool = True,
            range_clip: float = 0.0,
            pos_embedding_dim: int = 32,
            pos_embedding_denom: float = 10000.0,
            pos_embedding_max_len: int = 1000,
            decoder_prenet_n_layers: int = 2,
            decoder_prenet_dim: int = 256,
            decoder_prenet_activation: nn.Module = nn.ReLU(),
            decoder_prenet_p_dropout: float = 0.5,
            decoder_dim: int = 1024,
            decoder_n_layers: int = 1,
            output_dim: int = 128,
            postnet_num_convolutions: int = 5,
            postnet_hidden_features: int = 512,
            postnet_kernel_size: int = 5,
            postnet_activation: nn.Module = nn.Tanh()):
        super(Sodium, self).__init__()
        self.encoder = SodiumEncoder(
            num_lyrics, num_pitches,
            embedding_lyric_dim=embedding_lyric_dim,
            embedding_pitch_dim=embedding_pitch_dim,
            embedding_dim=embedding_dim,
            prenet_num_convolutions=encoder_prenet_num_convolutions,
            prenet_kernel_size=encoder_prenet_kernel_size,
            prenet_activation=encoder_prenet_activation,
            p_dropout=encoder_p_dropout,
            use_transformer=encoder_use_transformer,
            transformer_nlayers=encoder_transformer_nlayers,
            transformer_nhead=encoder_transformer_nhead,
            transformer_ff_dim=encoder_transformer_ff_dim,
            transformer_activation=encoder_transformer_activation,
            lstm_num_layers=encoder_lstm_num_layers,
            lstm_zoneout=encoder_lstm_zoneout)
        self.upsampler = SodiumUpsampler(
            embedding_dim=embedding_dim,
            duration_hidden_dim=duration_hidden_dim,
            duration_n_layers=duration_n_layers,
            duration_bias=duration_bias,
            range_hidden_dim=range_hidden_dim,
            range_n_layers=range_n_layers,
            range_bias=range_bias,
            range_clip=range_clip,
            pos_embedding_dim=pos_embedding_dim,
            pos_embedding_denom=pos_embedding_denom,
            pos_embedding_max_len=pos_embedding_max_len)
        self.decoder = SodiumDecoder(
            embedding_dim=embedding_dim + pos_embedding_dim,
            prenet_n_layers=decoder_prenet_n_layers,
            prenet_dim=decoder_prenet_dim,
            prenet_activation=decoder_prenet_activation,
            prenet_p_dropout=decoder_prenet_p_dropout,
            decoder_dim=decoder_dim,
            decoder_n_layers=decoder_n_layers,
            output_dim=output_dim)
        self.post_net = SodiumPostnet(
            num_convolutions=postnet_num_convolutions,
            features=output_dim,
            hidden_features=postnet_hidden_features,
            kernel_size=postnet_kernel_size,
            activation=postnet_activation)

    def forward(self, lyrics, pitches, durations, tempo, target_durations, input_lengths, mels):
        encoder_out = self.encoder(lyrics, pitches, input_lengths)
        pred_durations, upsampled, _, weights = self.upsampler(
            encoder_out, durations, tempo, target_durations, input_lengths)
        output = self.decoder(upsampled, mels)

        postnet_output = output.permute(1, 2, 0)
        postnet_output = self.post_net(postnet_output)
        postnet_output = postnet_output.permute(2, 0, 1)
        postnet_output = output + postnet_output

        return output, postnet_output, pred_durations, weights

    def infer(self, lyrics, pitches, durations, tempo):
        encoder_out = self.encoder.infer(lyrics, pitches)
        upsampled, _, weights = self.upsampler.infer(encoder_out, durations, tempo)
        output = self.decoder.infer(upsampled)

        if output.shape[0] > 1:
            postnet_output = output.permute(1, 2, 0)
            postnet_output = self.post_net(postnet_output)
            postnet_output = postnet_output.permute(2, 0, 1)
            postnet_output = output + postnet_output
        else:
            postnet_output = torch.zeros(output.shape)

        return output, postnet_output, weights
