import torch
from torch.nn import Module

class BoxCoxTransform(Module):
    def __init__(self, lambda_one, lambda_two=0):
        super(BoxCoxTransform, self).__init__()
        self.lambda_one = lambda_one
        self.lambda_two = lambda_two
        if self.lambda_one == 0:
            self._forward_fn = lambda y: torch.log(y + self.lambda_two)
        else:
            self._forward_fn = lambda y: (((y + self.lambda_two) ** self.lambda_one) - 1) / self.lambda_one

    def forward(self, y):
        return self._forward_fn(y)


class ZScoreTransform(Module):
    def __init__(self):
        super(ZScoreTransform, self).__init__()
    
    def forward(_, y):
        return (y - y.mean()) / y.std()
        