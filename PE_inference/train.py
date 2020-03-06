import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import glob
import numpy as np
import pickle as p
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset.custom_dataloader import ComposedImageData
from torch.utils.data import DataLoader
from model.drn import drn_d_105
from model.mymodel import MyModel
# from torchsummary import summary
from dataset.metrics import Metrics
from utils.losses import balanced_binary_cross_entropy, mse
from utils.utils import nms
from tqdm import tqdm
import torchvision.models as models

##############################################################################################################
############################################### Define Model #################################################
##############################################################################################################

drn = drn_d_105(pretrained=True, channels=3).cuda()
drn = nn.Sequential(*list(drn.children())[:-2]).cuda()
net = MyModel(drn).cuda()

# for params in list(net.parameters())[:-5]:
#     params.requires_grad = False

# for params in net.parameters():
#     print(params.requires_grad)
# summary(net, (4, 256, 256))

##############################################################################################################
############################################# Setup Training #################################################
##############################################################################################################

# define input folders
PREFIX = '/home/nelson/Workspace/'
RGB_FOLDER = '{}/cities_dataset/rgb'.format(PREFIX)
ANNOT_FOLDER = '{}/cities_dataset/annot'.format(PREFIX)

# define train/val lists
with open('/home/nelson/Workspace/cities_dataset/train_list.txt') as f:
    train_list = [line.strip() for line in f.readlines()]
with open('/home/nelson/Workspace/cities_dataset/valid_list.txt') as f:
    valid_list = [line.strip() for line in f.readlines()]

# create dataset
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
dset_train = ComposedImageData(RGB_FOLDER, ANNOT_FOLDER, train_list, mean, std, augment=True, depth_folder=None, gray_folder=None, surf_folder=None)
dset_val = ComposedImageData(RGB_FOLDER, ANNOT_FOLDER, valid_list, mean, std, augment=False, depth_folder=None, gray_folder=None, surf_folder=None)
dset_list = {'train': dset_train, 'val': dset_val}

# create loaders
train_loader = DataLoader(dset_train, batch_size=8, shuffle=True, num_workers=8)
valid_loader = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=8)
dset_loader = {'train': train_loader, 'val': valid_loader}
data_iterator = {'train': tqdm(train_loader, total=len(dset_train)), 'val': tqdm(valid_loader, total=len(dset_val))}

# select optimizer
optimizer = optim.Adam(filter(lambda x:x.requires_grad, net.parameters()), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
best_score = 9999999.0

##############################################################################################################
############################################### Start Training ###############################################
##############################################################################################################

for epoch in range(1000):

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            net.train()  # Set model to training mode
        else:
            net.eval()

        running_loss = 0.0
        for i, data in enumerate(data_iterator[phase]):
            # get the inputs
            xs, es = data
            xs, es = Variable(xs.float().cuda()), Variable(es.float().cuda())
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if phase == 'train':
                seg = net(xs)
            else:
                with torch.no_grad():
                    seg = net(xs)
            seg = F.sigmoid(seg.squeeze(1))
            seg = seg.squeeze(1)
            # compute losses
            l_seg = F.binary_cross_entropy(seg, es, size_average=True)
            # seg = seg.transpose(1, 3).contiguous().view(-1, 19)
            # es = es.view(-1)
            running_loss += l_seg

            # step
            if phase == 'train':
                l_seg.backward()
                optimizer.step()


        # print epoch loss
        print('[%d] %s lr: %f \nloss: %.5f' %
              (epoch + 1, phase, optimizer.param_groups[0]['lr'], running_loss / len(dset_loader[phase])))

        # tack best model
        if phase == 'val':
            
            # if running_loss < best_score:
            #     print('new best: loss %.5f' % running_loss)
            #     best_score = running_loss
            #     torch.save(net, './best_segmenter.pth')
            torch.save(net, './best_segmenter.pth')

        # reset running loss
        running_loss = 0.0

print('Finished Training')