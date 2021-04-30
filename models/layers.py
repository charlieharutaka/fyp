import warnings

import torch.nn as nn
from torch.nn.init import xavier_uniform_, kaiming_uniform_, calculate_gain


class Linear(nn.Linear):
    def __init__(self, *args, nonlinearity='linear', kaiming=False, leaky_relu_slope=None, kaiming_mode='fan_in', **kwargs):
        if kaiming and (nonlinearity != 'relu' or nonlinearity != 'leaky_relu'):
            warnings.warn(f"Using Kaiming initialization with {nonlinearity} nonlinearity is not recommended", UserWarning)

        super(Linear, self).__init__(*args, **kwargs)

        if kaiming:
            leaky_relu_slope = 1 if leaky_relu_slope is None else leaky_relu_slope
            kaiming_uniform_(self.weight, a=leaky_relu_slope, mode=kaiming_mode, nonlinearity=nonlinearity)
        else:
            xavier_uniform_(self.weight, calculate_gain(nonlinearity, param=leaky_relu_slope))


class Conv(nn.Conv1d):
    def __init__(self, *args, nonlinearity='linear', kaiming=False, leaky_relu_slope=None, kaiming_mode='fan_in', **kwargs):

        if kaiming and (nonlinearity != 'relu' or nonlinearity != 'leaky_relu'):
            warnings.warn(f"Using Kaiming initialization with {nonlinearity} nonlinearity is not recommended", UserWarning)

        super(Conv, self).__init__(*args, **kwargs)
        if kaiming:
            leaky_relu_slope = 1 if leaky_relu_slope is None else leaky_relu_slope
            kaiming_uniform_(self.weight, a=leaky_relu_slope, mode=kaiming_mode, nonlinearity=nonlinearity)
        else:
            xavier_uniform_(self.weight, calculate_gain(nonlinearity, param=leaky_relu_slope))
