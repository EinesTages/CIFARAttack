import pytorch_lightning as pl
import torch
import torch.nn as nn
from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
import os

all_classifiers = {
    "vgg11_bn": vgg11_bn(pretrained=True),
    "vgg13_bn": vgg13_bn(pretrained=True),
    "vgg16_bn": vgg16_bn(pretrained=True),
    "vgg19_bn": vgg19_bn(pretrained=True),
    "resnet18": resnet18(pretrained=True),
    "resnet34": resnet34(pretrained=True),
    "resnet50": resnet50(pretrained=True),
    "densenet121": densenet121(pretrained=True),
    "densenet161": densenet161(pretrained=True),
    "densenet169": densenet169(pretrained=True),
    "mobilenet_v2": mobilenet_v2(pretrained=True),
    "googlenet": googlenet(pretrained=True),
    "inception_v3": inception_v3(pretrained=True),
}


class CIFAR10Model(nn.Module):
    def __init__(self, classifier_name):
        super(CIFAR10Model, self).__init__()
        self.model = all_classifiers[classifier_name]

    def forward(self, x):
        x = self.model(x)
        return x
