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
        

class PowerToDecibelTransform(Module):
    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0):
        super(PowerToDecibelTransform, self).__init__()
        assert amin > 0, "amin must be strictly positive"
        if top_db is not None:
            assert top_db > 0, "top_db must be non-negative"
        if callable(ref):
            self._callable_ref = True
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def forward(self, S):
        if self._callable_ref:
            ref_value = self.ref(S)
        else:
            ref_value = torch.abs(torch.full_like(S, self.ref))
        amin_tensor = torch.full_like(S, self.amin)
        log_spec = 10.0 * torch.log10(torch.maximum(amin_tensor, S))
        log_spec = log_spec - (10.0 * torch.log10(torch.maximum(amin_tensor, ref_value)))

        if self.top_db is not None:
            log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)
        return log_spec


class ScaleToIntervalTransform(Module):
    def __init__(self, lower=0.0, upper=1.0):
        super(ScaleToIntervalTransform, self).__init__()
        assert upper > lower, "Interval must have positive width"
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        return self.lower + ((x - x.min()) * (self.upper - self.lower)) / (x.max() - x.min())


class DynamicRangeCompression(Module):
    def __init__(self, C=1, clip_val=1e-5):
        super(DynamicRangeCompression, self).__init__()
        self.C = C
        self.clip_val = clip_val
    
    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.clip_val) * self.C)
