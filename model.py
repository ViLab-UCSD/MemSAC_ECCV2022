import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np


__all__ = []


class Resnet(nn.Module):
  def __init__(self, resnet, pretrained=True):
    super().__init__()
    model_resnet = getattr(models, resnet)(pretrained=pretrained)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class AdversarialLayer(torch.autograd.Function):
    iter_num = 0
    max_iter = 12500
    @staticmethod
    def forward(ctx, input):
        #self.iter_num += 1
#         ctx.save_for_backward(iter_num, max_iter)
        AdversarialLayer.iter_num += 1
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        alpha = 10
        low = 0.0
        high = 1.0
        lamb = 2.0
        iter_num, max_iter = AdversarialLayer.iter_num, AdversarialLayer.max_iter 
        # print('iter_num {}'.format(iter_num))
        coeff = np.float(lamb * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
        return -coeff * gradOutput

  

# class Resnet34(nn.Module):
#   def __init__(self):
#     super(Resnet34, self).__init__()
#     model_resnet34 = models.resnet34(pretrained=True)
#     self.conv1 = model_resnet34.conv1
#     self.bn1 = model_resnet34.bn1
#     self.relu = model_resnet34.relu
#     self.maxpool = model_resnet34.maxpool
#     self.layer1 = model_resnet34.layer1
#     self.layer2 = model_resnet34.layer2
#     self.layer3 = model_resnet34.layer3
#     self.layer4 = model_resnet34.layer4
#     self.avgpool = model_resnet34.avgpool
#     self.__in_features = model_resnet34.fc.in_features

#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)
#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)
#     x = self.avgpool(x)
#     x = x.view(x.size(0), -1)
#     return x

#   def output_num(self):
#     return self.__in_features

# class Resnet50(nn.Module):
#   def __init__(self):
#     super(Resnet50, self).__init__()
#     model_resnet50 = models.resnet50(pretrained=True)
#     self.conv1 = model_resnet50.conv1
#     self.bn1 = model_resnet50.bn1
#     self.relu = model_resnet50.relu
#     self.maxpool = model_resnet50.maxpool
#     self.layer1 = model_resnet50.layer1
#     self.layer2 = model_resnet50.layer2
#     self.layer3 = model_resnet50.layer3
#     self.layer4 = model_resnet50.layer4
#     self.avgpool = model_resnet50.avgpool
#     self.__in_features = model_resnet50.fc.in_features

#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)
#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)
#     x = self.avgpool(x)
#     x = x.view(x.size(0), -1)
#     return x

#   def output_num(self):
#     return self.__in_features

# class Resnet101(nn.Module):
#   def __init__(self):
#     super(Resnet101, self).__init__()
#     model_resnet101 = models.resnet101(pretrained=True)
#     self.conv1 = model_resnet101.conv1
#     self.bn1 = model_resnet101.bn1
#     self.relu = model_resnet101.relu
#     self.maxpool = model_resnet101.maxpool
#     self.layer1 = model_resnet101.layer1
#     self.layer2 = model_resnet101.layer2
#     self.layer3 = model_resnet101.layer3
#     self.layer4 = model_resnet101.layer4
#     self.avgpool = model_resnet101.avgpool
#     self.__in_features = model_resnet101.fc.in_features

#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)
#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)
#     x = self.avgpool(x)
#     x = x.view(x.size(0), -1)
#     return x

#   def output_num(self):
#     return self.__in_features


# class Resnet152(nn.Module):
#   def __init__(self):
#     super(Resnet152, self).__init__()
#     model_resnet152 = models.resnet152(pretrained=True)
#     self.conv1 = model_resnet152.conv1
#     self.bn1 = model_resnet152.bn1
#     self.relu = model_resnet152.relu
#     self.maxpool = model_resnet152.maxpool
#     self.layer1 = model_resnet152.layer1
#     self.layer2 = model_resnet152.layer2
#     self.layer3 = model_resnet152.layer3
#     self.layer4 = model_resnet152.layer4
#     self.avgpool = model_resnet152.avgpool
#     self.__in_features = model_resnet152.fc.in_features

#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)
#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)
#     x = self.avgpool(x)
#     x = x.view(x.size(0), -1)
#     return x

#   def output_num(self):
#     return self.__in_features
