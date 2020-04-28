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
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
from dataset.custom_dataloader import ComposedImageData
from torch.utils.data import DataLoader
from dataset.metrics import Metrics
from utils.utils import nms, compose_im
from model.mymodel import MyModel
import os

##############################################################################################################
################################################ Load Model ##################################################
##############################################################################################################

net = torch.load('./best_segmenter.pth').cuda()
net = net.eval()

# define input folders
PREFIX = '/home/nelson/Workspace/'
RGB_FOLDER = '{}/cities_dataset/rgb'.format(PREFIX)
ANNOT_FOLDER = '{}/cities_dataset/annot'.format(PREFIX)

# define train/val lists
with open('{}/cities_dataset/all_list.txt'.format(PREFIX)) as f:
    valid_list = [line.strip() for line in f.readlines()]

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
dset_val = ComposedImageData(RGB_FOLDER, ANNOT_FOLDER, valid_list, mean, std, augment=False, depth_folder=None, gray_folder=None, surf_folder=None)
valid_loader = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=1)
mt = Metrics()

##############################################################################################################
############################################# Start Prediction ###############################################
##############################################################################################################
os.makedirs('./output', exist_ok=True)
for k, data in enumerate(valid_loader):

    # get the inputs
    xs, es = data
    xs, es = Variable(xs.float().cuda()), Variable(es.float().cuda())

    # run model
    seg = net(xs)
    seg = F.sigmoid(seg).squeeze(1).squeeze(0).detach().cpu().numpy()

    # draw output
    seg = (seg - np.min(seg))/(np.max(seg) - np.min(seg))
    seg = seg*255.0
    seg_im = Image.fromarray(np.array(seg).astype('uint8'))
    seg_im.save('./output/{}.jpg'.format(valid_list[k]))

