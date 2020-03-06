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
with open('/home/nelson/Workspace/cities_dataset/all_list.txt') as f:
    valid_list = [line.strip() for line in f.readlines()]

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
dset_val = ComposedImageData(RGB_FOLDER, ANNOT_FOLDER, valid_list, mean, std, augment=False, depth_folder=None, gray_folder=None, surf_folder=None)
valid_loader = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=1)
mt = Metrics()

##############################################################################################################
############################################# Start Prediction ###############################################
##############################################################################################################
for k, data in enumerate(valid_loader):

    # get the inputs
    xs, es = data
    xs, es = Variable(xs.float().cuda()), Variable(es.float().cuda())

    # run model
    seg = net(xs)
    seg = F.sigmoid(seg).squeeze(1).squeeze(0).detach().cpu().numpy()
    #seg = F.softmax(seg, -1).squeeze(0).squeeze(0).detach().cpu().numpy()
    #seg = np.argmax(seg, -1)
    # get input image
    # im_path = os.path.join(RGB_FOLDER, valid_list[k]+'.jpg')
    # im = Image.open(im_path)
    #print(seg.shape)
    # draw output
    seg = (seg - np.min(seg))/(np.max(seg) - np.min(seg))
    seg = seg*255.0
    #seg = np.array(es[0,: ,:])

    #print(seg)
    # for angle in range(1, 20):
    #     seg2 = np.array(es[0,: ,:])
    #     seg2[seg2 != angle] = 0
    #     seg2[seg2 == angle] = 255
        
    #     if np.sum(seg2) == 0:
    #         continue

    #     seg_im2 = Image.fromarray(np.array(seg2).astype('uint8'))
    #     seg_im = Image.fromarray(np.array(seg).astype('uint8'))

        # print(angle)
        # plt.imshow(seg_im2)
        # plt.figure()
        # plt.imshow(seg_im)
        # plt.show()

    seg_im = Image.fromarray(np.array(seg).astype('uint8'))

    #seg_im = compose_im(np.array(seg), seg)
    #seg = (seg - np.min(seg))/(np.max(seg) - np.min(seg))
    # plt.imshow(seg_im)
    # plt.show()

    #seg_im = seg_im.resize((512, 512))
 
    seg_im.save('{}/cities_dataset/edge_map/{}.jpg'.format(PREFIX, valid_list[k]))

