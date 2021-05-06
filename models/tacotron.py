import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout


class HighwayLayer(nn.Module):
    """
    Highway network described by:
    y = H(x) * T(x) + (1 - H(x)) * x
    """

    def __init__(self, features):
        super(HighwayLayer, self).__init__()
        self.features = features
        self.fc = nn.Linear(features, 2 * features)

    def forward(self, x):
        f, c = torch.split(self.fc(x), self.features, dim=-1)
        f = F.relu(f)
        c = torch.sigmoid(c)
        return c * f + (1 - c) * x


class ConvolutionBank(nn.Module):
    """
    Multiple parallel convolutions from 1 to num_ngrams.
    The result is concatenated together in the channel dimension
    hence result will have shape (batch, num_ngrams * num_channels, seq)
    """

    def __init__(self, num_ngrams, num_channels):
        super(ConvolutionBank, self).__init__()
        convolutions = []
        for kernel_size in range(1, num_ngrams + 1):
            conv = nn.Conv1d(
                num_channels,
                num_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1)
            torch.nn.init.xavier_uniform_(
                conv.weight,
                gain=torch.nn.init.calculate_gain('relu'))
            padding = nn.ConstantPad1d(
                (kernel_size // 2, (kernel_size - 1) // 2), 0.0)
            batch_norm = nn.BatchNorm1d(num_channels)
            convolutions.append(nn.Sequential(
                padding, conv, batch_norm, nn.ReLU()))
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x):
        out = []
        for conv in self.convolutions:
            c = conv(x)
            out.append(c)
        return torch.cat(out, dim=1)


class CBHG(nn.Module):
    """
    Convolution-Bank Highway GRU module.
    1. Conv1D Bank (K=16, C_k=128)
    2. MaxPool1D (stride=1, width=2)
    3. Conv1D Projections:
        1. Conv1D(kernel_size=3, out_channels=128) > ReLU
        1. Conv1D(kernel_size=3, out_channels=128)
    4. Highway Network (4 x Linear(128) > ReLU)
    5. Bidirectional GRU (128)
    """

    def __init__(self,
                 num_channels=128,
                 bank_size=16,
                 max_pool_width=2,
                 projection_kernel_size=3,
                 highway_size=4):
        super(CBHG, self).__init__()

        # Parameters
        self.num_channels = num_channels

        # Bank & Pool
        self.bank = ConvolutionBank(bank_size, num_channels)
        self.pool = nn.Sequential(
            nn.ConstantPad1d(
                (max_pool_width // 2, (max_pool_width - 1) // 2), float('-inf')),
            nn.MaxPool1d(max_pool_width, stride=1))

        # Projection
        conv1 = nn.Conv1d(num_channels * bank_size,
                          num_channels, projection_kernel_size)
        torch.nn.init.xavier_uniform_(
            conv1.weight,
            gain=torch.nn.init.calculate_gain('relu'))
        conv2 = nn.Conv1d(num_channels, num_channels, projection_kernel_size)
        torch.nn.init.xavier_uniform_(
            conv2.weight,
            gain=torch.nn.init.calculate_gain('linear'))
        self.projection = nn.Sequential(
            nn.ConstantPad1d((projection_kernel_size // 2,
                              (projection_kernel_size - 1) // 2), 0.0),
            conv1,
            nn.ReLU(),
            nn.ConstantPad1d((projection_kernel_size // 2,
                              (projection_kernel_size - 1) // 2), 0.0),
            conv2)

        # Highway
        self.highway = nn.Sequential(*[
            HighwayLayer(num_channels) for _ in range(highway_size)])

        # GRU
        self.gru = nn.GRU(
            input_size=num_channels,
            hidden_size=num_channels // 2,
            batch_first=True,
            bidirectional=True)

    def forward(self, x, sequence_lengths=None):
        # Expect batch first
        batch_size, seq_len, num_channels = x.size()
        assert num_channels == self.num_channels

        out = x.transpose(1, 2)
        # x now of shape (batch_size, num_channels, seq_len)
        out = self.bank(out)
        out = self.pool(out)
        # x now of shape (batch_size, num_channels * bank_size, seq_len)
        out = self.projection(out)
        # x now of shape (batch_size, num_channels, seq_len)
        out = out.transpose(1, 2)
        # x now of shape (batch_size, seq_len, num_channels)

        # residual connection
        out = out + x
        out = self.highway(out)
        # pack padded sequence
        if sequence_lengths is not None:
            out = nn.utils.rnn.pack_padded_sequence(
                out, sequence_lengths, batch_first=True)
        out, _ = self.gru(out)
        # pad packed sequence
        if sequence_lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True)
        # Out shape is (batch_size, seq_len, num_channels)
        return out


class Encoder(nn.Module):
    """
    TacoTron encoder.
    2. Pre-net:
        1. Linear(256) > ReLU > Dropout(0.5)
        2. Linear(256) > ReLU > Dropout(0.5)
    3. CBHG as above
    """

    def __init__(self, input_dim=256, output_dim=256, p_dropout=0.5):
        super(Encoder, self).__init__()
        self.prenet = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout))
        self.cbhg = CBHG(output_dim)

    def forward(self, x):
        out = self.prenet(x)
        out = self.cbhg(out)
        return out


class EncoderV2(nn.Module):
    """
    TacoTron2 Encoder.
    """

    def __init__(self, input_dim=256, output_dim=256, p_dropout=0.5, n_convolutions=3, kernel_size=5):
        super(EncoderV2, self).__init__()
        self.conv_banks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(input_dim if i == 0 else output_dim, output_dim, kernel_size,
                          padding=((kernel_size - 1) // 2)),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(p_dropout))
            for i in range(n_convolutions)
        ])
        self.lstm = nn.LSTM(input_size=output_dim, hidden_size=(
            output_dim // 2), batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        input_lengths = input_lengths.cpu().to(torch.int64)
        # x of shape (batch, sequence, features)
        x = self.conv_banks(x.transpose(1, 2))
        x = x.transpose(1, 2)
        # x of shape (batch, sequence, features)
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x

    def infer(self, x):
        # x of shape (1, sequence, features)
        x = self.conv_banks(x.transpose(1, 2))
        x = x.transpose(1, 2)
        # x of shape (batch, sequence, features)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class TemporalEncoder(nn.Module):
    """
    Encoder implementing rhythmic & pitch information into the output of the RNN.
    """

    def __init__(self, num_embeddings, lyric_dim=256, pitch_dim=256, rhythm_dim=256, output_dim=256, p_dropout=0.5, n_convolutions=3, kernel_size=5):
        super(TemporalEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, lyric_dim)
        self.pitch_proj = nn.Linear(1, pitch_dim)
        self.rhythm_proj = nn.Linear(1, rhythm_dim)

        all_dims = lyric_dim + pitch_dim + rhythm_dim
        self.conv_banks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(all_dims if i == 0 else output_dim, output_dim, kernel_size,
                          padding=((kernel_size - 1) // 2)),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(p_dropout))
            for i in range(n_convolutions)
        ])
        self.lstm = nn.LSTM(input_size=output_dim, hidden_size=(
            output_dim // 2), batch_first=True, bidirectional=True)

    def forward(self, lyrics, pitches, rhythms, input_lengths):
        input_lengths = input_lengths.cpu().to(torch.int64)
        # all (batch, seq)
        lyrics = self.embedding(lyrics)
        pitches = self.pitch_proj(pitches.to(torch.float).unsqueeze(-1))
        rhythms = self.rhythm_proj(rhythms.unsqueeze(-1))
        # all (batch, feature, seq)
        output = torch.cat([lyrics, pitches, rhythms], dim=-1).transpose(1, 2)
        output = self.conv_banks(output).transpose(1, 2)
        self.lstm.flatten_parameters()
        output = nn.utils.rnn.pack_padded_sequence(
            output, input_lengths, batch_first=True)
        output, _ = self.lstm(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output

    def infer(self, lyrics, pitches, rhythms):
        # all (batch, seq)
        lyrics = self.embedding(lyrics)
        pitches = self.pitch_proj(pitches.to(torch.float).unsqueeze(-1))
        rhythms = self.rhythm_proj(rhythms.unsqueeze(-1))
        # all (batch, feature, seq)
        output = torch.cat([lyrics, pitches, rhythms], dim=-1).transpose(1, 2)
        output = self.conv_banks(output).transpose(1, 2)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(output)
        return output


def get_mask_from_lengths(lengths):
    # Given a length tensor, create masks
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=lengths.new_empty(
        (max_len,), dtype=torch.long))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class LocationLayer(nn.Module):
    """
    Location Sensitive Attention: https://arxiv.org/pdf/1506.07503v1.pdf
    Uses convolutional filters to do something idk
    """

    def __init__(self,
                 n_filters,
                 kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1)
        self.fc = nn.Linear(
            n_filters,
            attention_dim,
            bias=False)

    def forward(self, attention):
        attention = self.conv(attention)
        attention = attention.transpose(1, 2)
        attention = self.fc(attention)
        return attention


class AttentionLayer(nn.Module):
    """
    Additive (Bahdanau) Attention: https://arxiv.org/pdf/1409.0473.pdf
    """

    def __init__(
            self,
            attention_dim,
            attention_rnn_dim,
            attention_location_n_filters,
            attention_location_kernel_size):
        super(AttentionLayer, self).__init__()

        self.query = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.value = nn.Linear(attention_dim, 1, bias=False)
        self.location = LocationLayer(
            attention_location_n_filters, attention_location_kernel_size, attention_dim)
        self.score_mask_value = -float("inf")

    def forward(
            self,
            attention_hidden_state,
            memory,
            processed_memory,
            attention_weights_cat,
            mask=None):
        # Calculate alignment energies
        processed_query = self.query(attention_hidden_state.unsqueeze(1))
        processed_attention_weights = self.location(attention_weights_cat)
        alignment = self.value(torch.tanh(
            processed_query + processed_memory + processed_attention_weights))
        alignment = alignment.squeeze(-1)

        # Mask data for padded sequences
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        # Calculate attention
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Postnet(nn.Module):
    """
    Sequential 1-d convolutions to improve the quality of the mel-spectrogram
    """

    def __init__(self, hidden_dim=256, n_convolutions=5, embedding_dim=512, kernel_size=5, p_dropout=0.5):
        super(Postnet, self).__init__()
        padding = (kernel_size - 1) // 2
        convolutions = [nn.Sequential(
            nn.Conv1d(hidden_dim if i == 0 else embedding_dim, hidden_dim if i ==
                      n_convolutions - 1 else embedding_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim if i == n_convolutions -
                           1 else embedding_dim),
            nn.Tanh() if i == n_convolutions - 1 else nn.Identity(),
            nn.Dropout(p_dropout)
        ) for i in range(n_convolutions)]
        self.convolutions = nn.Sequential(*convolutions)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.convolutions(x).transpose(1, 2)


class DecoderCell(nn.Module):
    def __init__(self,
                 num_encoders=1,
                 embedding_dim=256,
                 attention_dim=128,
                 prenet_dim=128,
                 attention_rnn_dim=1024,
                 p_attention_dropout=0.1,
                 attention_location_n_filters=32,
                 attention_location_kernel_size=31,
                 decoder_rnn_dim=1024,
                 p_decoder_dropout=0.1,
                 hidden_dim=256):
        super(DecoderCell, self).__init__()
        # Keep params
        self.num_encoders = num_encoders  # number of encoded streams
        self.hidden_dim = hidden_dim  # n_mel_channels
        self.embedding_dim = embedding_dim  # encoder_embedding_dim
        self.attention_dim = attention_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.attention_location_n_filters = attention_location_n_filters
        self.attention_location_kernel_size = attention_location_kernel_size
        self.decoder_rnn_dim = decoder_rnn_dim
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.prenet_dim = prenet_dim
        # Layers
        # Memory layer to process encoder outputs
        self.memories_layer = nn.ModuleList(
            [nn.Linear(embedding_dim, attention_dim, bias=False) for _ in range(self.num_encoders)])
        # Attention modules
        self.attention_rnns = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.attention_dropout = nn.Dropout(p_attention_dropout)
        for _ in range(self.num_encoders):
            self.attention_rnns.append(nn.LSTMCell(
                embedding_dim + prenet_dim, attention_rnn_dim))
            self.attention_layers.append(AttentionLayer(
                attention_dim,
                attention_rnn_dim,
                attention_location_n_filters,
                attention_location_kernel_size))
        # Decoder modules
        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + embedding_dim, decoder_rnn_dim)
        self.decoder_dropout = nn.Dropout(p_decoder_dropout)
        # Projection to the mel-spectrogram space
        self.linear_projection = nn.Linear(
            decoder_rnn_dim + embedding_dim, hidden_dim)
        # Gating layer that learns to stop
        self.gate = nn.Linear(decoder_rnn_dim + embedding_dim, 1)

    def init_go_frame(self, encoder_out):
        # Initializes the 0 frame signalling the start of the sample
        batch_size = encoder_out[0].shape[0]
        return encoder_out[0].new_zeros((batch_size, self.hidden_dim), requires_grad=True)

    def init_rnn_states(self, encoder_out, masks):
        # Initialize all the RNN hidden states
        batch_size, max_steps, _ = encoder_out[0].shape
        # Attention hidden states
        self.attention_hidden = [encoder_out[i].new_zeros(
            (batch_size, self.attention_rnn_dim), requires_grad=True) for i in range(self.num_encoders)]
        self.attention_cell = [encoder_out[i].new_zeros(
            (batch_size, self.attention_rnn_dim), requires_grad=True) for i in range(self.num_encoders)]
        # Attention weights
        self.attention_weights = [encoder_out[i].new_zeros(
            (batch_size, max_steps), requires_grad=True) for i in range(self.num_encoders)]
        self.attention_weights_cum = [encoder_out[i].new_zeros(
            (batch_size, max_steps), requires_grad=True) for i in range(self.num_encoders)]
        self.attention_context = [encoder_out[i].new_zeros(
            (batch_size, self.embedding_dim)) for i in range(self.num_encoders)]
        # Decoder hidden states
        self.decoder_hidden = encoder_out[0].new_zeros(
            (batch_size, self.decoder_rnn_dim), requires_grad=True)
        self.decoder_cell = encoder_out[0].new_zeros(
            (batch_size, self.decoder_rnn_dim), requires_grad=True)

        self.memories = encoder_out
        self.processed_memories = []
        for i, memory in enumerate(encoder_out):
            processed_memory = self.memories_layer[i](memory)
            self.processed_memories.append(processed_memory)
        self.masks = masks

    def forward(self, last_frame):
        # Do attention for each of the encoder streans
        for i in range(self.num_encoders):
            # Concatenate the previous decoder frame with the current attention context in the channel dimension
            attention_in = torch.cat(
                [last_frame, self.attention_context[i]], dim=-1)
            attention_hidden, attention_cell = self.attention_rnns[i](
                attention_in, (self.attention_hidden[i], self.attention_cell[i]))
            self.attention_cell[i] = attention_cell
            self.attention_hidden[i] = self.attention_dropout(attention_hidden)
            # Concatenate the current and cumulative attention weights in the channel(?) dimension
            attention_weights = torch.cat([
                self.attention_weights[i].unsqueeze(1),
                self.attention_weights_cum[i].unsqueeze(1)], dim=1)
            attention_context, attention_weights = self.attention_layers[i](
                self.attention_hidden[i], self.memories[i], self.processed_memories[i],
                attention_weights, self.masks[i])
            self.attention_context[i] = attention_context
            self.attention_weights[i] = attention_weights
            # Update the cumulative attention weights
            self.attention_weights_cum[i] = self.attention_weights_cum[i] + \
                self.attention_weights[i]
        # Combine the hidden and context states by addition
        attention_hidden = 0
        for h in self.attention_hidden:
            attention_hidden = attention_hidden + h
        attention_context = 0
        for c in self.attention_context:
            attention_context = attention_context + c
        # Concatenate the attention hidden state and attention context in the channel dimension
        decoder_in = torch.cat([attention_hidden, attention_context], dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_in, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = self.decoder_dropout(self.decoder_hidden)
        # Produce the next frame by concatenating the hidden state and attention context and using the projection layer
        decoder_hidden_attention_context = torch.cat([
            self.decoder_hidden, attention_context], dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        # Get a stopping prediction
        stop_prediction = self.gate(decoder_hidden_attention_context)

        return decoder_output, stop_prediction, torch.stack(self.attention_weights, dim=1)


class Decoder(nn.Module):
    def __init__(
            self,
            num_encoders=1,
            hidden_dim=256,
            embedding_dim=256,
            attention_dim=128,
            attention_rnn_dim=1024,
            attention_location_n_filters=32,
            attention_location_kernel_size=31,
            decoder_rnn_dim=1024,
            p_prenet_dropout=0.1,
            p_attention_dropout=0.1,
            p_decoder_dropout=0.1,
            prenet_dim=128,
            max_decoder_steps=1000,
            stopping_threshold=0.5):
        super(Decoder, self).__init__()
        # Keep params
        self.num_encoders = num_encoders
        self.hidden_dim = hidden_dim  # n_mel_channels
        self.embedding_dim = embedding_dim  # encoder_embedding_dim
        self.attention_dim = attention_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.attention_location_n_filters = attention_location_n_filters
        self.attention_location_kernel_size = attention_location_kernel_size
        self.decoder_rnn_dim = decoder_rnn_dim
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.stopping_threshold = stopping_threshold
        # Layers
        # Prenet applied to each RNN state before passing it through the RNN
        self.pre_net = nn.Sequential(
            nn.Linear(hidden_dim, prenet_dim),
            nn.ReLU(),
            nn.Dropout(p_prenet_dropout),
            nn.Linear(prenet_dim, prenet_dim),
            nn.ReLU(),
            nn.Dropout(p_prenet_dropout))
        # Attention modules
        self.cell = DecoderCell(
            num_encoders=num_encoders,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            prenet_dim=prenet_dim,
            attention_rnn_dim=attention_rnn_dim,
            p_attention_dropout=p_attention_dropout,
            attention_location_n_filters=attention_location_n_filters,
            attention_location_kernel_size=attention_location_kernel_size,
            decoder_rnn_dim=decoder_rnn_dim,
            p_decoder_dropout=p_decoder_dropout,
            hidden_dim=hidden_dim)

    def forward(self, encoder_out, decoder_inputs, memory_lengths_list):
        """
        Assume the shapes:
        - encoder_out (num_encoders, batch, time_in, embedding_dim)
        - decoder_inputs (batch, time_out, hidden_dim)
        - memory_lengths_list (num_encoders, time_in,)
        """
        # Seq-first order for RNNs
        go_frame = self.cell.init_go_frame(encoder_out).unsqueeze(0)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        # decoder_inputs shape is now (time_out, batch, hidden_dim)
        decoder_inputs = torch.cat((go_frame, decoder_inputs), dim=0)
        decoder_inputs = self.pre_net(decoder_inputs)

        self.cell.init_rnn_states(
            encoder_out, masks=[~get_mask_from_lengths(memory_lengths) for memory_lengths in memory_lengths_list])

        outputs, stops, alignments = [], [], []
        while len(outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(outputs)]
            output, stop, attn_weights = self.cell(decoder_input)
            outputs.append(output.squeeze(1))
            stops.append(stop.squeeze(1))
            alignments.append(attn_weights)
        # Shape of outputs is (time_out, batch, hidden_dim)
        outputs = torch.stack(outputs).transpose(
            0, 1).contiguous()  # to (batch, time, hidden_dim)
        # Shape of stops is (time_out, batch)
        stops = torch.stack(stops).permute(
            1, 0).contiguous()  # to (batch, time)
        # Shape of alignments is (time_out, batch)
        alignments = torch.stack(alignments).permute(
            2, 1, 0, 3).contiguous()  # to (num_encoders, batch, time_out, time_in)

        return outputs, stops, alignments

    def infer(self, encoder_out):
        """
        Perform inference
        Assume the shapes:
        - encoder_out (num_encoders, batch, time_in, embedding_dim)
        """
        decoder_input = self.cell.init_go_frame(encoder_out)
        self.cell.init_rnn_states(encoder_out, None)

        outputs, stops, alignments = [], [], []
        for i in range(self.max_decoder_steps):
            decoder_input = self.pre_net(decoder_input)
            output, stop, alignment = self.cell(decoder_input)
            outputs.append(output.squeeze(1))
            stops.append(stop.squeeze(1))
            alignments.append(alignment)

            if torch.sigmoid(stop).item() > self.stopping_threshold:
                break

            decoder_input = output
        if i == self.max_decoder_steps - 1:
            warnings.warn("Max decoder steps reached", UserWarning)
        # Shape of outputs is (time_out, batch, hidden_dim)
        outputs = torch.stack(outputs).transpose(
            0, 1).contiguous()  # to (batch, time, hidden_dim)
        # Shape of stops is (time_out, batch)
        stops = torch.stack(stops).permute(
            1, 0).contiguous()  # to (batch, time)
        # Shape of alignments is (time_out, batch)
        alignments = torch.stack(alignments).permute(
            2, 1, 0, 3).contiguous()  # to (num_encoders, batch, time_out, time_in)

        return outputs, stops, alignments


class Tacotron(nn.Module):
    def __init__(self,
                 num_embeddings,
                 encoder_lyric_dim=256,
                 encoder_pitch_dim=256,
                 encoder_rhythm_dim=256,
                 embedding_dim=256,
                 encoder_n_convolutions=3,
                 encoder_kernel_size=5,
                 encoder_p_dropout=0.5,
                 hidden_dim=256,
                 attention_dim=128,
                 attention_rnn_dim=1024,
                 attention_location_n_filters=32,
                 attention_location_kernel_size=31,
                 decoder_rnn_dim=1024,
                 p_prenet_dropout=0.1,
                 p_attention_dropout=0.1,
                 p_decoder_dropout=0.1,
                 prenet_dim=128,
                 max_decoder_steps=1000,
                 stopping_threshold=0.5,
                 postnet_n_convolutions=5,
                 postnet_embedding_dim=512,
                 postnet_kernel_size=5,
                 postnet_p_dropout=0.5):
        super(Tacotron, self).__init__()
        self.lyric_embedding = nn.Embedding(
            num_embeddings, embedding_dim=encoder_lyric_dim)
        self.note_embedding = nn.Embedding(
            128, embedding_dim=encoder_pitch_dim)
        self.rhythm_embedding = nn.Linear(1, encoder_rhythm_dim)
        self.lyric_encoder = EncoderV2(input_dim=encoder_lyric_dim,
                                       output_dim=embedding_dim,
                                       p_dropout=encoder_p_dropout,
                                       n_convolutions=encoder_n_convolutions,
                                       kernel_size=encoder_kernel_size)
        self.note_encoder = EncoderV2(input_dim=encoder_pitch_dim,
                                      output_dim=embedding_dim,
                                      p_dropout=encoder_p_dropout,
                                      n_convolutions=encoder_n_convolutions,
                                      kernel_size=encoder_kernel_size)
        self.rhythm_encoder = EncoderV2(input_dim=encoder_rhythm_dim,
                                        output_dim=embedding_dim,
                                        p_dropout=encoder_p_dropout,
                                        n_convolutions=encoder_n_convolutions,
                                        kernel_size=encoder_kernel_size)
        # self.encoder = TemporalEncoder(num_embeddings,
        #                                lyric_dim=encoder_lyric_dim,
        #                                pitch_dim=encoder_pitch_dim,
        #                                rhythm_dim=encoder_rhythm_dim,
        #                                output_dim=embedding_dim,
        #                                p_dropout=encoder_p_dropout,
        #                                n_convolutions=encoder_n_convolutions,
        #                                kernel_size=encoder_kernel_size)
        self.decoder = Decoder(num_encoders=3,
                               hidden_dim=hidden_dim,
                               embedding_dim=embedding_dim,
                               attention_dim=attention_dim,
                               attention_rnn_dim=attention_rnn_dim,
                               attention_location_n_filters=attention_location_n_filters,
                               attention_location_kernel_size=attention_location_kernel_size,
                               decoder_rnn_dim=decoder_rnn_dim,
                               p_prenet_dropout=p_prenet_dropout,
                               p_attention_dropout=p_attention_dropout,
                               p_decoder_dropout=p_decoder_dropout,
                               prenet_dim=prenet_dim,
                               max_decoder_steps=max_decoder_steps,
                               stopping_threshold=stopping_threshold)
        self.postnet = Postnet(hidden_dim=hidden_dim,
                               n_convolutions=postnet_n_convolutions,
                               embedding_dim=postnet_embedding_dim,
                               kernel_size=postnet_kernel_size,
                               p_dropout=postnet_p_dropout)

    def forward(self, lyric_inputs, note_inputs, duration_inputs, input_lengths, mels):
        """
        Teacher-forcing training.

        Shapes:
        - note_inputs (batch, sequence)
        - lyric_inputs (batch, sequence)
        - input_lengths (batch,)
        - mels (batch, time, channels)
        """
        # embedded_notes = self.note_embedding(
        #     note_inputs.to(torch.float).unsqueeze(-1))
        # embedded_inputs = self.embedding(lyric_inputs)
        # encoder_inputs = torch.cat(
        #     [embedded_notes, embedded_inputs], dim=-1).transpose(1, 2)
        # encoder_outputs = self.encoder(encoder_inputs, input_lengths)
        # encoder_outputs = self.encoder(
        #     lyric_inputs, note_inputs, duration_inputs, input_lengths)

        embedded_lyrics = self.lyric_embedding(lyric_inputs)
        embedded_notes = self.note_embedding(note_inputs)
        embedded_rhythm = self.rhythm_embedding(duration_inputs.unsqueeze(-1))
        encoded_lyrics = self.lyric_encoder(embedded_lyrics, input_lengths)
        encoded_notes = self.lyric_encoder(embedded_notes, input_lengths)
        encoded_rhythm = self.lyric_encoder(embedded_rhythm, input_lengths)

        outputs, stops, alignments = self.decoder(
            [encoded_lyrics, encoded_notes, encoded_rhythm], mels, [input_lengths, input_lengths, input_lengths])
        outputs_postnet = self.postnet(outputs)
        outputs_postnet = outputs + outputs_postnet

        return outputs, outputs_postnet, stops, alignments

    def infer(self, lyric_inputs, note_inputs, duration_inputs):
        """
        Inference.

        Shapes:
        - note_inputs (1, sequence)
        - lyric_inputs (1, sequence)
        """
        # embedded_notes = self.note_embedding(
        #     note_inputs.to(torch.float).unsqueeze(-1))
        # embedded_inputs = self.embedding(lyric_inputs)
        # encoder_inputs = torch.cat(
        #     [embedded_notes, embedded_inputs], dim=-1).transpose(1, 2)
        # encoder_outputs = self.encoder.infer(encoder_inputs)

        embedded_lyrics = self.lyric_embedding(lyric_inputs)
        embedded_notes = self.note_embedding(note_inputs)
        embedded_rhythm = self.rhythm_embedding(duration_inputs.unsqueeze(-1))
        encoded_lyrics = self.lyric_encoder.infer(embedded_lyrics)
        encoded_notes = self.lyric_encoder.infer(embedded_notes)
        encoded_rhythm = self.lyric_encoder.infer(embedded_rhythm)

        outputs, stops, alignments = self.decoder.infer([encoded_lyrics, encoded_notes, encoded_rhythm])
        outputs_postnet = self.postnet(outputs)
        outputs_postnet = outputs + outputs_postnet

        return outputs, outputs_postnet, stops, alignments
