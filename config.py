import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

REPRESENTATION_NAMES = ['keypoints3d','depth_euclidean']

FC_NEURON_LISTS = [8*len(REPRESENTATION_NAMES)*16*16,1024,1024,8*len(REPRESENTATION_NAMES)*16*16]
RESIDUAL_LAYERS_PER_BLOCK = [2,2,2,2]
RESIDUAL_SIZE   = [32, 64, 128, 256]
RESIDUAL_NEURON_CHANNEL   = [16, 8, 4, 2, 2]
STRIDES = [1, 1, 1]






BATCHSIZE = 4


