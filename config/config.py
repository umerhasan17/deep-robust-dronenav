import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'
EXPERIMENT_IDS = ['Baseline', 'BaselineMidLevel', 'DRRN', 'DRRNActualMap','DRRNSupervisedMap']

EXPERIMENT_ID_INDEX = 2
CURRENT_POLICY = EXPERIMENT_IDS[EXPERIMENT_ID_INDEX]

REPRESENTATION_NAMES = ['keypoints3d', 'depth_euclidean']

FC_NEURON_LISTS = [8 * len(REPRESENTATION_NAMES) * 16 * 16, 1024, 1024, 8 * len(REPRESENTATION_NAMES) * 16 * 16]
RESIDUAL_LAYERS_PER_BLOCK = [2, 2, 2, 2]
RESIDUAL_SIZE = [32, 64, 128, 256]
if EXPERIMENT_ID_INDEX >= 2:
    RESIDUAL_NEURON_CHANNEL = [16, 8, 4, 2, 2]
else:
    RESIDUAL_NEURON_CHANNEL = [16, 8, 4, 2, 3]
STRIDES = [1, 1, 1]
IMG_DIMENSIONS = (3, 256, 256)  # mid level reps are in colour right now
MAP_DIMENSIONS = (2, 256, 256)
MAP_DOWNSAMPLE = 2 ** 3
BATCHSIZE = 4

""" Config to create image map dataset for supervised training of mapper architecture + RL architecture. """
DATASET_SAVE_PERIOD = 20
DATASET_SAVE_FOLDER = 'data/image_map_dataset/'
START_IMAGE_NUMBER = 0

MAP_SIZE = (5, 5)  # map size (in [m]), given a 256x256 map, picking map size = 5 gives a resolution of ~2cm

# TODO add habitat_config yaml path
# TODO add data path for use in habitat_config

HABITAT_CONFIGS_PATH = 'configs/'

