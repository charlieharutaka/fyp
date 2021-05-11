import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.nn.init import xavier_uniform_, kaiming_uniform_, calculate_gain


class Linear(nn.Linear):
    def __init__(
            self,
            *args,
            nonlinearity='linear',
            kaiming=False,
            leaky_relu_slope=None,
            kaiming_mode='fan_in',
            **kwargs):
        if kaiming and (nonlinearity != 'relu' or nonlinearity != 'leaky_relu'):
            warnings.warn(
                f"Using Kaiming initialization with {nonlinearity} nonlinearity is not recommended",
                UserWarning)

        super(Linear, self).__init__(*args, **kwargs)

        if kaiming:
            leaky_relu_slope = 1 if leaky_relu_slope is None else leaky_relu_slope
            kaiming_uniform_(self.weight, a=leaky_relu_slope, mode=kaiming_mode, nonlinearity=nonlinearity)
        else:
            xavier_uniform_(self.weight, calculate_gain(nonlinearity, param=leaky_relu_slope))


class Conv(nn.Conv1d):
    def __init__(
            self,
            *args,
            nonlinearity='linear',
            kaiming=False,
            leaky_relu_slope=None,
            kaiming_mode='fan_in',
            **kwargs):

        if kaiming and (nonlinearity != 'relu' or nonlinearity != 'leaky_relu'):
            warnings.warn(
                f"Using Kaiming initialization with {nonlinearity} nonlinearity is not recommended",
                UserWarning)

        super(Conv, self).__init__(*args, **kwargs)
        if kaiming:
            leaky_relu_slope = 1 if leaky_relu_slope is None else leaky_relu_slope
            kaiming_uniform_(self.weight, a=leaky_relu_slope, mode=kaiming_mode, nonlinearity=nonlinearity)
        else:
            xavier_uniform_(self.weight, calculate_gain(nonlinearity, param=leaky_relu_slope))


class ZoneOutLSTM(nn.Module):
    """
    ZoneOut LSTM
    https://arxiv.org/pdf/1606.01305.pdf
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.0,
            zoneout: float = 0.0,
            bidirectional: bool = False):
        # proj_size: int = 0):
        super(ZoneOutLSTM, self).__init__()
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # Forward direction
        self.cells = nn.ModuleList()
        self.cells.append(nn.LSTMCell(input_size, hidden_size, bias))
        for _ in range(num_layers - 1):
            self.cells.append(nn.LSTMCell(hidden_size, hidden_size, bias))
        # Reverse direction
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        if self.bidirectional:
            self.cells_rev = nn.ModuleList()
            self.cells_rev.append(nn.LSTMCell(input_size, hidden_size, bias))
            for _ in range(num_layers - 1):
                self.cells_rev.append(nn.LSTMCell(hidden_size, hidden_size, bias))
        # Dropout & zoneout probs
        self.p_dropout = dropout
        self.p_zoneout = zoneout
        self.dropout = D.Bernoulli(1.0 - dropout)
        self.zoneout = D.Bernoulli(1.0 - zoneout)
        # Projection
        # self.use_projection = proj_size > 0
        # if self.use_projection:
        #     self.projection = nn.Linear(hidden_size, proj_size)
        # Hidden/cell states

    def reset_hidden_states(self):
        self.hidden_states = []

    def init_hidden_states(self, inputs):
        _, batch, _ = inputs.shape
        self.hidden_states.append((
            inputs.new_zeros(
                self.num_directions *
                self.num_layers,
                batch,
                self.hidden_size,
                requires_grad=False),
            inputs.new_zeros(
                self.num_directions *
                self.num_layers,
                batch,
                self.hidden_size,
                requires_grad=False)))

    def forward(self, inputs: torch.Tensor, init_states=None):
        """
        Args:
            inputs: (sequence, batch, input_size)
            init_states: (h0, c0)
        Returns:
            outputs: (sequence, batch, num_directions * hidden_size)
            final_states: (hn, cn)
        """
        self.reset_hidden_states()
        if init_states:
            self.hidden_states.append(init_states)
        else:
            self.init_hidden_states(inputs)
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        max_length, _, _ = inputs.shape

        outputs = []
        rev_outputs = []

        # Iterate over input frames
        for idx in range(max_length):
            # Get the old hidden states
            old_hidden_states = self.hidden_states[idx][0]
            old_cell_states = self.hidden_states[idx][1]
            new_hidden_states = []
            new_cell_states = []
            # Forward
            output = inputs[idx]
            for layer in range(self.num_layers):
                hidden = old_hidden_states.narrow(0, layer, 1).squeeze(0)
                cell =  old_cell_states.narrow(0, layer, 1).squeeze(0)
                new_hidden, new_cell = self.cells[layer](output, (hidden, cell))
                # Apply dropout on output
                output = new_hidden
                if self.p_dropout > 0 and layer < self.num_layers - 1:
                    dropout = self.dropout.sample(new_hidden.shape).to(new_hidden)
                    output = output * dropout
                # Append states
                new_hidden_states.append(new_hidden)
                new_cell_states.append(new_cell)
            outputs.append(output)
            # Reverse
            if self.bidirectional:
                output = inputs[(max_length - 1) - idx]
                for layer in range(self.num_layers):
                    hidden = old_hidden_states.narrow(0, self.num_layers + layer, 1).squeeze(0)
                    cell = old_cell_states.narrow(0, self.num_layers + layer, 1).squeeze(0)
                    new_hidden, new_cell = self.cells[layer](output, (hidden, cell))
                    # Apply dropout on output
                    dropout = self.dropout.sample(new_hidden.shape).to(new_hidden)
                    output = new_hidden * dropout
                    # Append states
                    new_hidden_states.append(new_hidden)
                    new_cell_states.append(new_cell)
                rev_outputs.append(output)
            new_hidden_states = torch.stack(new_hidden_states, dim=0)
            new_cell_states = torch.stack(new_cell_states, dim=0)
            # Apply zoneout
            if self.p_zoneout > 0:
                zoneout_h = self.zoneout.sample(new_hidden_states.shape).to(new_hidden_states)
                zoneout_c = self.zoneout.sample(new_cell_states.shape).to(new_cell_states)
                new_hidden_states = new_hidden_states * zoneout_h + (old_hidden_states * (1 - zoneout_h))
                new_cell_states = new_cell_states * zoneout_c + (old_cell_states * (1 - zoneout_c))
            self.hidden_states.append((new_hidden_states, new_cell_states))
            
        outputs = torch.stack(outputs, dim=0)
        if self.bidirectional:
            rev_outputs = torch.stack(rev_outputs[::-1], dim=0)
            outputs = torch.cat((outputs, rev_outputs), dim=2)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, self.hidden_states[-1]

