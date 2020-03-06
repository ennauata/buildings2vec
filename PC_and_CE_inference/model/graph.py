import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# GNN - Graph Neural Network
class EdgeClassifier(nn.Module):
    def __init__(self, base):
        super(EdgeClassifier, self).__init__()
        self.base = base
        self.fc1 = nn.Linear(1000, 512, bias=True)
        self.fc2 = nn.Linear(512, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # apply backbone
        x = self.relu(self.base(x))

        # classifier head
        x = self.relu(self.fc1(x))
        z = F.sigmoid(self.fc2(x)).view(-1)

        return  z
