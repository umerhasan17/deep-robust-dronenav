import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

REPRESENTATION_NAMES = ['keypoints3d','depth_euclidean']

FC_NEURON_LISTS = [8*len(REPRESENTATION_NAMES)*16*16,8,8,8]

BATCHSIZE = 4