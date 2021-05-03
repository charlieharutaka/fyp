from pprint import pprint
from utils.musicxml import parse_musicxml
from utils.datasets import VocalSetDataset

dataset = VocalSetDataset(n_fft=800, n_mels=128, rebuild_cache=True)
print(dataset[0])