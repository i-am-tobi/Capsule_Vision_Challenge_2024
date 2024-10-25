import torch
import torch.nn as nn
import torch.nn.functional as F
from CNN_Model import DenseNet
from projection import ProjectionHead
from MLP import MLP
from Classifier_Head import ClassificationHead


class Model(nn.Module):
    def __init__(self, num_numerical_features:int, num_classes:int,
                 growth_rate: int, num_layers: int, reduction: float = 0.5,
                 bottleneck: bool = True, use_dropout: bool = True):

        super(Model, self).__init__()

        self.cnn = DenseNet(growth_rate, num_layers, reduction, bottleneck, use_dropout)  # Example using ResNet18
        self.proj = ProjectionHead(self.cnn.out_channels,out_dim=512)

        self.mlp = MLP(num_numerical_features,embedding_dim=512)

        self.cls = ClassificationHead(num_class=num_classes)


    def forward(self, image, mask, numerical_data):

        cnn_features = self.cnn(image, mask)[0]
        cnn_features = self.proj(cnn_features)

        mlp_features = self.mlp(numerical_data)

        combined_features = torch.cat((cnn_features, mlp_features), dim=1)

        output = self.cls(combined_features)

        return output