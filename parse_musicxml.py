from pprint import pprint
from utils.musicxml import parse_musicxml
from utils.datasets import VocalSetDataset

dataset = VocalSetDataset()
print(dataset[0])