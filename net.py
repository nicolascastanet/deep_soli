import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#from torch.optim import *

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = nn.Sequential(
                            nn.Conv3d(4, 32, kernel_size=(3, 3, 3), padding=0),
                            nn.ReLU())
        
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(32*64*3*3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.batch=nn.BatchNorm1d(128)
        # Dropout
        self.drop=nn.Dropout(p=0.2)
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        import ipdb; ipdb.set_trace()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch(x)
        x = self.drop(x)
        x = self.fc2(x)
        
        return x