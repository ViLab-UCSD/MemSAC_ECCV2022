import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np


class Resnet(nn.Module):
  def __init__(self, resnet_n, pretrained=True):
    super().__init__()
    model_resnet = getattr(models, resnet_n)(pretrained=pretrained)
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

class discriminator(nn.Module):
    def __init__(self, feature_len, total_classes):
        super().__init__()

        self.ad_layer1 = nn.Linear(feature_len * total_classes, 1024)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)
        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3)

    def forward(self, x, y):
        op_out = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))
        ad_in = op_out.view(-1, y.size(1) * x.size(1))
        f2 = self.fc1(ad_in)
        f = self.fc2_3(f2)
        return f


class Classifier(nn.Module):
    def __init__(self, feature_len, cate_num):
        super().__init__()
        self.classifier = nn.Linear(feature_len, cate_num)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, features):
        activations = self.classifier(features)
        return (activations)


class Encoder(nn.Module):
    def __init__(self, resnet, bn_dim=256, total_classes=None):
        super(Encoder, self).__init__()
        self.model_fc = Resnet(resnet)
        feature_len = self.model_fc.output_num()
        self.bottleneck_0 = nn.Linear(feature_len, bn_dim)
        self.bottleneck_0.weight.data.normal_(0, 0.005)
        self.bottleneck_0.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_0, nn.BatchNorm1d(bn_dim), nn.ReLU())
        self.total_classes = total_classes
        if total_classes:
            self.classifier_layer = Classifier(bn_dim, total_classes)

    def forward(self, x):
        features = self.model_fc(x)
        out_bottleneck = self.bottleneck_layer(features)
        if not self.total_classes:
            return (out_bottleneck, None)
        logits = self.classifier_layer(out_bottleneck)
        return (out_bottleneck, logits)

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
