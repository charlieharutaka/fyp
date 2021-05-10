import torch
import matplotlib.pyplot as plt

from models.sodium import Sodium

model = Sodium(52, 128, encoder_use_transformer=False)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))

a = model(torch.randint(0, 52, (32, 8)), torch.randint(0, 52, (32, 8)), torch.ones((32, 8)), torch.ones((8,)), torch.full((32, 8), 4), torch.full((8,), 32), torch.randn((128, 8, 128)), torch.full((8,), 128))
print(a[0].shape)

a = model.infer(torch.randint(0, 52, (32, 8)), torch.randint(0, 52, (32, 8)), torch.ones((32, 8)), torch.ones((8,)))
print(a[0].shape)