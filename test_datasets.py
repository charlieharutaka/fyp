import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import VocalSetDataset
dataset = VocalSetDataset(
    n_fft=800,
    n_mels=192,
    f_min=80.0,
    f_max=8000.0,
    exclude=['spoken'],
    rebuild_cache=False,
    transpose_steps=[-1, 0, 1])
print(dataset[0])

# writer = SummaryWriter('runs/test')
# a = torch.load('data\\VocalSet\\cache\\female1_arpeggios_c_slow_belt_a_-1.pt')
# b = torch.load('data\\VocalSet\\cache\\female1_arpeggios_c_slow_belt_a_0.pt')
# c = torch.load('data\\VocalSet\\cache\\female1_arpeggios_c_slow_belt_a_+1.pt')

# writer.add_audio('-1', a.wave, sample_rate=16000)
# writer.add_audio('0', b.wave, sample_rate=16000)
# writer.add_audio('+1', c.wave, sample_rate=16000)

# print(a.notes)
# print(b.notes)
# print(c.notes)

# fig, ax = plt.subplots(3, 1)
# ax[0].imshow(torch.log(a.mel).transpose(0, 1), aspect='auto',
#           origin='lower',
#           interpolation='nearest',
#           cmap='magma')
# ax[1].imshow(torch.log(b.mel).transpose(0, 1), aspect='auto',
#           origin='lower',
#           interpolation='nearest',
#           cmap='magma')
# ax[2].imshow(torch.log(c.mel).transpose(0, 1), aspect='auto',
#           origin='lower',
#           interpolation='nearest',
#           cmap='magma')
# plt.show()
# writer.close()
