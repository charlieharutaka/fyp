import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionQueue(nn.Module):
    """
    The convolution queue as described in the fast-wavenet algorithm

    Args:
    - {int} num_channels:   Number of channels
    - {int} max_length:     Max length of the queue
    
    """

    def __init__(self, num_channels, max_length):
        super(ConvolutionQueue, self).__init__()
        self.num_channels = num_channels
        self.max_length = max_length
        self.deque = deque(maxlen=max_length)
        self.clear_queue()
    
    def forward(self, x):
        # Enqueue x and dequeue the last item
        assert x.shape[0] == self.num_channels, "Number of channels do not match"

        popped = self.deque.pop()
        self.deque.appendleft(x)

        return popped
    
    def clear_queue(self, device=torch.device('cpu')):
        self.deque.clear()
        for _ in range(self.max_length):
            self.deque.appendleft(torch.zeros((self.num_channels,), device=device))


class GatedActivationUnit(nn.Module):
    """
    Gated Activation Unit

    Args:
    - {int} in_channels:                Number of input channels
    - {int} out_channels:               Number of output channels
    - {int | (int, int)} kernel_size:   Size of convolving kernel
    - {(int | (int, int))?} stride:     Kernel stride
    - {(int | (int, int))?} padding:    Padding on data
    - {(int | (int, int))?} dilation:   Kernel dilation
    - {int?} groups:                    Number of kernel groups
    - {bool?} bias:                     Whether to use bias or not

    Forward pass:
    - input: (N, in_channels, L)
    - output: (N, out_channels, L')
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(GatedActivationUnit, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels,
                           2 * out_channels,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           bias=bias)
    
    def forward(self, x):
        f, g = torch.split(self.conv(x), self.out_channels, dim=1)
        f = torch.tanh(f)
        g = torch.sigmoid(g)
        return f * g


class ResidualBlock(nn.Module):
    """
    Residual Block with Skip Connections

    Args:
    - {int} in_channels:                Number of input channels
    - {int} dilation_channels:          Number of dilation channels
    - {int} residual_channels:          Number of residual channels
    - {int} skip_channels:              Number of skip channels
    - {int | (int, int)} kernel_size:   Size of convolving kernel
    - {(int | (int, int))?} stride:     Kernel stride
    - {(int | (int, int))?} padding:    Padding on data
    - {(int | (int, int))?} dilation:   Kernel dilation
    - {int?} groups:                    Number of kernel groups
    - {bool?} bias:                     Whether to use bias or not

    Forward pass:
    - input: (N, in_channels, L)
    - output: (N, residual_channels, L'), (N, skip_channels, L')
    """
    def __init__(self,
               in_channels,
               dilation_channels,
               residual_channels,
               skip_channels,
               kernel_size,
               stride=1,
               padding=0, 
               dilation=1,
               groups=1,
               bias=True):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv = GatedActivationUnit(in_channels,
                                        dilation_channels,
                                        kernel_size, 
                                        stride=stride,
                                        padding=padding, 
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias)
        self.resd_conv = nn.Conv1d(dilation_channels,
                                   residual_channels,
                                   1,
                                   stride=1,
                                   padding=0,
                                   dilation=1,
                                   groups=1,
                                   bias=bias)
        self.skip_conv = nn.Conv1d(dilation_channels,
                                   skip_channels,
                                   1,
                                   stride=1,
                                   padding=0,
                                   dilation=1,
                                   groups=1,
                                   bias=bias)

    def forward(self, x):
        out = self.conv(x)
        res = self.resd_conv(out)
        skip = self.skip_conv(out)
        out = x[:, :, (self.kernel_size - 1):] + res
        return out, skip


class DilatedQueueResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 initial_dilation=1,
                 groups=1,
                 bias=True):
        super(DilatedQueueResidualBlock, self).__init__()
        self.dilation = dilation
        self.initial_dilation = initial_dilation
        self.dilation_factor = int(dilation)# / initial_dilation)
        self.padding = padding
        self.residual_block = ResidualBlock(in_channels,
                                            dilation_channels,
                                            residual_channels,
                                            skip_channels,
                                            kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            dilation=1,
                                            groups=groups,
                                            bias=bias)
        self.queue = ConvolutionQueue(residual_channels, dilation)
    
    def forward(self, x):
        # If in training, use dilation without queues
        if self.training:
            # Shape is (N, C, L)
            x, left_pad = self.dilate(x, self.dilation_factor)
            x, skip = self.residual_block(x)
            x = self.undilate(x, self.dilation_factor, left_pad, self.padding)
            skip = self.undilate(skip, self.dilation_factor, left_pad, self.padding)
            return x, skip
        # If not training, use queues
        else:
            # Shape is (1, C, 1)
            # Assume x is a singular data point
            popped = self.queue(x.squeeze())
            popped = popped.unsqueeze(1).unsqueeze(0)
            # Concatenate popped and x to feed into conv
            x = torch.cat((popped, x), dim=2)
            # now shape is (1, C, 2)
            return self.residual_block(x)
    
    def dilate(self, x, dilation, left_pad=True):
        """
        Dilates the input along the last dimension

        Args:
        - {int} dilation:   Dilation size

        Forward pass:
        - input: (N, C, L)
        - output: (dilation, C, L * N / dilation)

        """
        if dilation == 1:
            return x, 0

        [N, C, L] = x.shape

        # Calculate if we need to pad the input
        new_L = int(math.ceil(L / dilation) * dilation)
        padding = 0
        if new_L != L:
            padding = new_L - L
            L = new_L
            if left_pad:
                x = F.pad(x, (padding, 0))
            else:
                x = F.pad(x, (0, padding))
        
        L = math.ceil(L / dilation)
        N = math.ceil(N * dilation)

        x = x.permute(1, 2, 0).reshape((C, L, N))
        x = x.permute(2, 0, 1)

        return x.contiguous(), padding

    def undilate(self, x, dilation, left_unpad, symm_unpad):
        """
        Reconstruct a (N * dilation, C, L // dilation) tensor
            into a (N, C, L) tensor
        """
        N, C, L = x.shape
        new_N = N // dilation
        new_L = L * dilation

        x = x.permute(1, 2, 0).reshape((C, new_L, new_N))
        x = x.permute(2, 0, 1)

        new_L = new_L - left_unpad - 2 * symm_unpad
        x = x.narrow(-1, left_unpad + symm_unpad, new_L)

        return x.contiguous()

    def clear_queue(self, device=torch.device('cpu')):
        self.queue.clear_queue(device)


class WaveNet(nn.Module):
    def __init__(self,
                 layers=10,
                 blocks=4,
                 channels=1,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 classes=256,
                 kernel_size=2,
                 bias=False):
        super(WaveNet, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.channels = channels
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size

        self.dilations = []
        self.receptive_field = 1

        self.input_conv = nn.Conv1d(in_channels=channels,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        self.residual_blocks = nn.ModuleList()
        for b in range(blocks):
            scope = kernel_size - 1
            dilation = 1
            initial_dilation = 1
            for l in range(layers):
                self.dilations.append((dilation, initial_dilation))
                self.residual_blocks.append(DilatedQueueResidualBlock(
                        in_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        residual_channels=residual_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        initial_dilation=initial_dilation,
                        bias=bias))
                self.receptive_field += scope
                scope *= 2
                initial_dilation = dilation
                dilation *=2
        
        self.output_conv = nn.Sequential(*[
                                          nn.Conv1d(in_channels=skip_channels,
                                                    out_channels=end_channels,
                                                    kernel_size=1,
                                                    bias=True),
                                          nn.ReLU(),
                                          nn.Conv1d(in_channels=end_channels,
                                                    out_channels=classes,
                                                    kernel_size=1,
                                                    bias=True)])

    def forward(self, x):
        x = self.input_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            x, s = self.residual_blocks[i](x)
            try:
                skip = skip.narrow(2, -s.size(2), s.size(2))
            except:
                skip = 0
            skip = s + skip
        
        x = F.relu(skip)
        x = self.output_conv(x)

        return x
    
    def clear_queues(self, device=torch.device('cpu')):
        for block in self.residual_blocks:
            block.clear_queue(device)