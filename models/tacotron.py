import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return out


class Encoder(nn.Module):
    """
    TacoTron encoder.
    1. Embedding (256)
    2. Pre-net:
        1. Linear(256) > ReLU > Dropout(0.5)
        2. Linear(256) > ReLU > Dropout(0.5)
    3. CBHG as above
    """

    def __init__(self, embedding_dim=256, p_dropout=0.5):
        super(Encoder, self).__init__()
        self.prenet = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout))
        self.cbhg = CBHG(embedding_dim)

    def forward(self, x):
        out = self.prenet(x)
        out = self.cbhg(out)
        return out


def get_mask_from_lengths(lengths, max_length=None):
    # Given a length tensor, create masks
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
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
        self.location_conv = nn.Conv1d(
            in_channels=2,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1)
        self.location_dense = nn.Linear(
            n_filters,
            attention_dim,
            bias=False)

    def forward(self, attention):
        attention = self.location_conv(attention)
        attention = attention.transpose(1, 2)
        attention = self.location_dense(attention)
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
        return self.convolutions(x)


class Decoder(nn.Module):
    def __init__(
            self,
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
        # Memory layer to process encoder outputs
        self.memory_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        # Prenet applied to each RNN state before passing it through the RNN
        self.pre_net = nn.Sequential(
            nn.Linear(hidden_dim, prenet_dim),
            nn.ReLU(),
            nn.Dropout(p_prenet_dropout),
            nn.Linear(prenet_dim, prenet_dim),
            nn.ReLU(),
            nn.Dropout(p_prenet_dropout))
        # Attention modules
        self.attention_rnn = nn.LSTMCell(
            embedding_dim + prenet_dim, attention_rnn_dim)
        self.attention_dropout = nn.Dropout(p_attention_dropout)
        self.attention_layer = AttentionLayer(
            attention_dim,
            attention_rnn_dim,
            attention_location_n_filters,
            attention_location_kernel_size)
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
        batch_size = encoder_out.shape[0]
        return encoder_out.new_zeros((batch_size, self.hidden_dim), requires_grad=True)

    def init_rnn_states(self, encoder_out, mask):
        # Initialize all the RNN hidden states
        batch_size, max_steps, _ = encoder_out.shape
        # Attention hidden states
        self.attention_hidden = encoder_out.new_zeros(
            (batch_size, self.attention_rnn_dim), requires_grad=True)
        self.attention_cell = encoder_out.new_zeros(
            (batch_size, self.attention_rnn_dim), requires_grad=True)
        # Decoder hidden states
        self.decoder_hidden = encoder_out.new_zeros(
            (batch_size, self.decoder_rnn_dim), requires_grad=True)
        self.decoder_cell = encoder_out.new_zeros(
            (batch_size, self.decoder_rnn_dim), requires_grad=True)
        # Attention weights
        self.attention_weights = encoder_out.new_zeros(
            (batch_size, max_steps), requires_grad=True)
        self.attention_weights_cum = encoder_out.new_zeros(
            (batch_size, max_steps), requires_grad=True)
        self.attention_context = encoder_out.new_zeros(
            (batch_size, self.embedding_dim))

        self.memory = encoder_out
        self.processed_memory = self.memory_layer(encoder_out)
        self.mask = mask

    def decode(self, last_frame):
        # Concatenate the previous decoder frame with the current attention context in the channel dimension
        attention_in = torch.cat([last_frame, self.attention_context], dim=-1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            attention_in, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = self.attention_dropout(self.attention_hidden)
        # Concatenate the current and cumulative attention weights in the channel(?) dimension
        attention_weights = torch.cat([
            self.attention_weights.unsqueeze(1),
            self.attention_weights_cum.unsqueeze(1)], dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights, self.mask)
        # Update the cumulative attention weights
        self.attention_weights_cum = self.attention_weights_cum + self.attention_weights
        # Concatenate the attention hidden state and attention context in the channel dimension
        decoder_in = torch.cat([
            self.attention_hidden, self.attention_context], dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_in, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = self.decoder_dropout(self.decoder_hidden)
        # Produce the next frame by concatenating the hidden state and attention context and using the projection layer
        decoder_hidden_attention_context = torch.cat([
            self.decoder_hidden, self.attention_context], dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        # Get a stopping prediction
        stop_prediction = self.gate(decoder_hidden_attention_context)

        return decoder_output, stop_prediction, self.attention_weights

    def forward(self, encoder_out, decoder_inputs, memory_lengths):
        """
        Assume the shapes:
        - encoder_out (batch, embedding_dim, time_in)
        - decoder_inputs (batch, hidden_dim, time_out)
        - memory_lengths (time_in,)
        """
        # Seq-first order for RNNs
        go_frame = self.init_go_frame(encoder_out).unsqueeze(0)
        decoder_inputs = decoder_inputs.permute(2, 0, 1)
        # decoder_inputs shape is now (time_out, batch, hidden_dim)
        decoder_inputs = torch.cat((go_frame, decoder_inputs), dim=0)
        decoder_inputs = self.pre_net(decoder_inputs)

        self.init_rnn_states(
            encoder_out, mask=~get_mask_from_lengths(memory_lengths))

        outputs, stops, alignments = [], [], []
        while len(outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(outputs)]
            output, stop, attn_weights = self.decode(
                decoder_input)
            outputs.append(output.squeeze(1))
            stops.append(stop.squeeze(1))
            alignments.append(attn_weights)
        # Shape of outputs is (time_out, batch, hidden_dim)
        outputs = torch.stack(outputs).permute(
            1, 2, 0).contiguous()  # to (batch, hidden_dim, time)
        # Shape of stops is (time_out, batch)
        stops = torch.stack(stops).permute(
            1, 0).contiguous()  # to (batch, time)
        # Shape of alignments is (time_out, batch)
        alignments = torch.stack(alignments).permute(
            1, 0, 2).contiguous()  # to (batch, time_out, time_in)

        return outputs, stops, alignments

    def infer(self, encoder_out):
        """
        Perform inference
        Assume the shapes:
        - encoder_out (batch, embedding_dim, time_in)
        """
        decoder_input = self.init_go_frame(encoder_out)
        self.init_rnn_states(encoder_out, None)

        outputs, stops, alignments = [], [], []
        for i in range(self.max_decoder_steps):
            decoder_input = self.pre_net(decoder_input)
            output, stop, alignment = self.decode(decoder_input)
            outputs.append(output.squeeze(1))
            stops.append(stop)
            alignments.append(alignment)

            if torch.sigmoid(stop.item()) > self.stopping_threshold:
                break

            decoder_input = output
        if i == self.max_decoder_steps - 1:
            warnings.warn("Max decoder steps reached", UserWarning)

        # Shape of outputs is (time_out, batch, hidden_dim)
        outputs = torch.stack(outputs).permute(
            1, 2, 0).contiguous()  # to (batch, hidden_dim, time)
        # Shape of stops is (time_out, batch)
        stops = torch.stack(stops).permute(
            1, 0).contiguous()  # to (batch, time)
        # Shape of alignments is (time_out, batch, time_in)
        alignments = torch.stack(alignments).permute(
            1, 0, 2).contiguous()  # to (batch, time_out, time_in)

        return outputs, stops, alignments


class Tacotron(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim=256,
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
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim=embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim,
                               p_dropout=encoder_p_dropout)
        self.decoder = Decoder(hidden_dim=hidden_dim,
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

    def forward(self, text_inputs, text_lengths, mels):
        """
        Teacher-forcing training.

        Shapes:
        - text_inputs (batch, sequence)
        - text_lengths list
        - mels (batch, channels, time)
        - max_len int
        - output_lengths list
        """
        embedded_inputs = self.embedding(text_inputs)
        encoder_outputs = self.encoder(embedded_inputs)
        outputs, stops, alignments = self.decoder(
            encoder_outputs, mels, text_lengths)
        outputs_postnet = self.postnet(outputs)
        outputs_postnet = outputs + outputs_postnet

        return outputs, outputs_postnet, stops, alignments
