import torch
import torch.utils.model_zoo
from config.config import REPRESENTATION_NAMES, DEBUG
from mapper.mid_level.encoder import mid_level_representations  # mid_level wrapper class


def encode_with_mid_level(image):

    image = torch.swapaxes(image, 1, 3)
    if DEBUG:
        print(f"Encoding image of shape {image.shape} with mid level encoders.")
    image = mid_level_representations(image, REPRESENTATION_NAMES)
    # (BATCH SIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor
    if DEBUG:
        print(f'Returning encoded representation of shape {image.shape}.')
    return image


def convert_rgb_obs_to_map(observations, fc_network, decoder_network):
    """
        Converts RGB tensor to whatever is outputted by the mapper architecture.
        In the full experiment, we train the mapper to output a map with a free space dimension and a confidence
        dimension.
    """

    assert 'rgb' in observations

    image = observations["rgb"]

    image = encode_with_mid_level(image)

    if DEBUG:
        print(f"Passing activation of shape {image.shape} through fcn.")

    image = image.view(image.shape[0], 1, -1)
    # flatten all dimensions except batch,
    # --> tensor of the form (BATCHSIZE x 2048*REPRESENTATION_NUMBER)

    image = fc_network(image)
    # pass through dense layer --> (BATCHSIZE x 2048*REPRESENTATION_NUMBER) tensor
    image = image.view(image.shape[0], 8 * len(REPRESENTATION_NAMES), 16, 16)

    # after fully connected layer, #(BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

    if DEBUG:
        print(f"Passing activation of shape {image.shape} to decoder.")

    decoder_output = decoder_network(image)
    decoder_output = torch.swapaxes(decoder_output, 1, 3)

    if DEBUG:
        print(f'Returning output tensor with shape: {decoder_output.shape}')

    observations["rgb"] = decoder_output

    return observations
