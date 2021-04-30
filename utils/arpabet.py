import torch

ARPABET_VOWELS = [
    "AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY",
    "EH", "ER", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW", "IX"]
ARPABET_CONSONANTS = [
    "B", "CH", "D", "DH", "DX", "EL", "EM", "EN", "F", "G", "HH", "H", "JH", "K",
    "L", "M", "N", "NG", "NX", "P", "Q", "R", "S", "SH", "T", "TH", "V", "W", "WH", "Y", "Z", "ZH"]
ARPABET = ARPABET_VOWELS + ARPABET_CONSONANTS


class ArpabetEncoding(object):
    def __init__(self, onehot=False):
        super(ArpabetEncoding, self).__init__()
        self.onehot = onehot
        self.arpa2idx = dict(zip(ARPABET, range(len(ARPABET))))
        self.idx2arpa = dict(zip(self.arpa2idx.values(), self.arpa2idx.keys()))
    
    def encode(self, arpa):
        idx = self.arpa2idx[arpa]
        if self.onehot:
            vec = torch.zeros((len(self.arpa2idx),))
            vec[idx] = 1.0
            return vec
        else:
            return idx
    
    def decode(self, idx):
        if self.onehot:
            idx = idx.nonzero().item()
        return self.idx2arpa[idx]
