import pdb

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from matplotlib.pyplot import imshow, imsave
import os
from torchvision import datasets, transforms
from torchvision.io import read_image
from mid_level.supervised_training_model import SupervisedTrainingModel
from mapper.mid_level.encoder import mid_level_representations  # mid_level wrapper class
import torch
from config.config import REPRESENTATION_NAMES, device
import torch.nn as nn
import torchvision.transforms.functional as TF
from os.path import normpath

torch.backends.cudnn.benchmark = True
batch_size = 2# 16 #TODO
num_workers =0 #TODO
num_epochs = 100
learning_rate = 1e-3


class CustomImageDataset(Dataset):
    def __init__(self,img_dir):
        self.img_dir = img_dir

    def __len__(self):
        return len(os.listdir(self.img_dir))//2

    def __getitem__(self, idx):
        rgb_path = normpath(os.path.join(self.img_dir,"rgb{}.jpeg".format(idx)))
        rgb = read_image(rgb_path).float()/256*2-1
        #rgb = TF.to_tensor(rgb) * 2 - 1

        map_path = normpath(os.path.join(self.img_dir,"map{}.jpeg".format(idx)))
        real_map = read_image(map_path).float()/256*2-1
        #real_map = TF.to_tensor(real_map) * 2 - 1

        return rgb, real_map
dataset = CustomImageDataset("./mapper/rgb_map_dataset")
train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
model = SupervisedTrainingModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

iters=0
train_loss = 0
for epoch in range(num_epochs):
    for rgb,real_map in train_loader:
        model.train()
        #pdb.set_trace()
        iters +=1
        rgb = rgb.to(device)
        real_map = real_map.to(device)
        # ===================forward=====================
        #rgb = torch.transpose(rgb, 1, 3)
        print(rgb.shape)
        activation = rgb
        # ==========Mid level encoder==========
        print("Passing mid level encoder...")
        activation = mid_level_representations(activation,REPRESENTATION_NAMES)  #  (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor
        # ==========FC+RESNET DECODER==========
        print("Passing FC+RESNET DECODER...")
        map_update = model(activation)
        #output = torch.transpose(map_update, 1, 3)
        print("Calculate loss and update...")
        loss = criterion(map_update, real_map)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if iters % 2 == 0:
            print(loss.item())
            with torch.no_grad():
                model.eval()
                map_update= map_update*0.5+0.5 # scale to (0,1)
                pic = torch.permute(map_update,(0,2,3,1)).cpu().data.numpy()[0]
                imsave( './mapper/debug_output/map_predict{}.png'.format(iters), pic)

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.item()))