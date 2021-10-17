from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess

def mid_level_representation(input_image_tensor, representation_name):
    """
    :return: pytorch tensor for 1 representation
    """
    pass

def concat_representation(input_image_tensor, representation_names):
    """
    :param input_image_tensor:
    :param representation_names:
    :return: concatted image tensor to pass into FCN
    """
    pass




if __name__ == '__main__':
    # # Download a test image
    # subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)

    # Load image and rescale/resize to [-1,1] and 3x256x256
    image = Image.open('test.png')
    x = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
    x = x.unsqueeze_(0)

    feature = 'keypoints3d'
    device = 'cpu'
    # Transform to normals feature
    representation = visualpriors.representation_transform(x, feature, device=device)

    # Transform to normals feature and then visualize the readout
    pred = visualpriors.feature_readout(x, feature, device=device)

    # Save it
    TF.to_pil_image(pred[0] / 2. + 0.5).save(f'test_{feature}_readout.png')
