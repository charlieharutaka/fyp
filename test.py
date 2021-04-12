from utils.datasets import ChoralSingingDataset

csd = ChoralSingingDataset('data', 4093)
print(len(csd))
print(csd.data[0][0].shape)
print(csd.data[0][1].shape)
print(csd.data[0][0][:,4093:] - csd.data[0][1][:,:799])
# print(csd.data[0][0].shape)
# print(csd.melspec(csd.data[0][0]))
