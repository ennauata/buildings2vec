import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, drn_base):
        super(MyModel, self).__init__()
        self.drn_base = drn_base
        self.bn0 = nn.BatchNorm2d(512, eps=0.001)
        self.bn1 = nn.BatchNorm2d(128, eps=0.001)
        self.bn2 = nn.BatchNorm2d(128, eps=0.001)
        self.bn3 = nn.BatchNorm2d(128, eps=0.001)
        self.bn4 = nn.BatchNorm2d(128, eps=0.001)
        self.bn5 = nn.BatchNorm2d(128, eps=0.001)
        self.bn6 = nn.BatchNorm2d(128, eps=0.001)

        self.deconv1 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)

        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 1, 1, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # extract features
        x = self.drn_base(x)  
        x = self.bn0(x)
        x = self.relu(x)

        # upsample
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # upsample
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # upsample
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # compute segments
        x = self.conv1(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn6(x)
        x = self.relu(x)

        seg = self.conv4(x)

        return seg