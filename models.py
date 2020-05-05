import torch, torch.nn as nn
from functions import ReverseLayerF
from torch.nn import functional as F

class LinearModel(nn.Module):

    def __init__(self, input_dim1, input_dim2, drop=0.25):
        super(LinearModel, self).__init__()
        
        # feature encoder1
        self.f1 = nn.Linear(input_dim1, 100)
        self.f1_drop = nn.Dropout(drop)

        # feature encoder2
        self.g1 = nn.Linear(input_dim2, 100)
        self.g1_drop = nn.Dropout(drop)
        
        # decoder
        self.d1 = nn.Linear(100, 100)
        self.d2 = nn.Linear(100, input_dim2)
        self.d_drop = nn.Dropout(drop)
        
        # sentiment classifier
        self.sc1 = nn.Linear(200, 10)
        self.sc2 = nn.Linear(10, 2)
        self.sc_drop = nn.Dropout(drop)
        
        # domain classifier
        self.dc1 = nn.Linear(200, 10)
        self.dc2 = nn.Linear(10, 2)
        self.dc_drop = nn.Dropout(drop)
        
    def encode1(self, x1):
        x1 = self.f1(x1)
        x1 = F.relu(x1)
        x1 = self.f1_drop(x1)
        return x1

    def encode2(self, x2):
        x2 = self.g1(x2)
        x2 = F.relu(x2)
        x2 = self.g1_drop(x2)
        return x2

    def decode(self, z):
        z = self.d1(z)
        z = F.relu(z)
        z = self.d_drop(z)
        # z = torch.sigmoid(self.d2(z))
        z = self.d2(z)
        return z 
    
    def domain_classifier(self, h):
        h = self.dc1(h)
        # h = self.dc_drop(h)
        h = F.relu(h)
        h = self.dc2(h)
        return h

    def sentiment_classifier(self, h):
        h = self.sc1(h)
        h = self.sc_drop(h)
        h = F.relu(h)
        h = self.sc2(h)
        return h

    def forward(self, input_data1, input_data2, alpha):
        
        z1 = self.encode1(input_data1)
        z2 = self.encode2(input_data2)
        
        reconstructed = self.decode(z2)

        z = torch.cat([z1, z2], axis=1)        

        reverse_z = ReverseLayerF.apply(z, alpha)
        class_output = self.sentiment_classifier(z)
        domain_output = self.domain_classifier(reverse_z)

        return reconstructed, class_output, domain_output