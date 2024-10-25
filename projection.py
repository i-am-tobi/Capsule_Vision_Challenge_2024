import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super(ProjectionHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_dim, 1024) 
        self.bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x