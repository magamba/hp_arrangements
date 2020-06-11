# -*- coding: utf-8 -*-

"""Model definitions
"""
import torch
import torch.nn as nn
from torchvision.models import vgg as VGG
from torchvision.models import alexnet as AlexNet

__all__ = [ 'lenet', 'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg19_orig', 'vgg11bn', 'vgg13bn', 'vgg16bn', 'vgg19bn']

def load_model(model_name, classes=1000, pretrained=True, in_channels=3):
  """Load the specified VGG architecture for ImageNet
  
    Args:
      model_name: VGG architecture type
      classes: number of predicted classes
      pretrained: load pretrained network on ImageNet
  """
  if pretrained:
    assert classes == 1000, "Pretrained models are provided only for Imagenet."
  
  kwargs = {'num_classes' : classes}
  
  if model_name == 'vgg11':
    net = VGG.vgg11(pretrained=pretrained, **kwargs)
    if in_channels != 3:
      input_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
      nn.init.kaiming_normal_(input_layer.weight, mode='fan_out', nonlinearity='relu')
      input_layer.bias.data.zero_()
      net.features[0] = input_layer
  elif model_name == 'vgg13':
    net = VGG.vgg13(pretrained=pretrained, **kwargs)
    if in_channels != 3:
      input_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
      nn.init.kaiming_normal_(input_layer.weight, mode='fan_out', nonlinearity='relu')
      input_layer.bias.data.zero_()
      net.features[0] = input_layer
  elif model_name == 'vgg16':
    net = VGG.vgg16(pretrained=pretrained, **kwargs)
    if in_channels != 3:
      input_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
      nn.init.kaiming_normal_(input_layer.weight, mode='fan_out', nonlinearity='relu')
      input_layer.bias.data.zero_()
      net.features[0] = input_layer
  elif model_name == 'vgg19':
    net = VGG.vgg19(pretrained=pretrained, **kwargs)
    if in_channels != 3:
      input_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
      nn.init.kaiming_normal_(input_layer.weight, mode='fan_out', nonlinearity='relu')
      input_layer.bias.data.zero_()
      net.features[0] = input_layer
  elif model_name == 'vgg11bn':
    net = VGG.vgg11_bn(pretrained=pretrained, **kwargs)
    if in_channels != 3:
      input_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
      nn.init.kaiming_normal_(input_layer.weight, mode='fan_out', nonlinearity='relu')
      input_layer.bias.data.zero_()
      net.features[0] = input_layer
  elif model_name == 'vgg13bn':
    net = VGG.vgg13_bn(pretrained=pretrained, **kwargs)
    if in_channels != 3:
      input_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
      nn.init.kaiming_normal_(input_layer.weight, mode='fan_out', nonlinearity='relu')
      input_layer.bias.data.zero_()
      net.features[0] = input_layer
  elif model_name == 'vgg16bn':
    net = VGG.vgg16_bn(pretrained=pretrained, **kwargs)
    if in_channels != 3:
      input_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
      nn.init.kaiming_normal_(input_layer.weight, mode='fan_out', nonlinearity='relu')
      input_layer.bias.data.zero_()
      net.features[0] = input_layer
  elif model_name == 'vgg19bn':
    net = VGG.vgg19_bn(pretrained=pretrained, **kwargs)
    if in_channels != 3:
      input_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
      nn.init.kaiming_normal_(input_layer.weight, mode='fan_out', nonlinearity='relu')
      input_layer.bias.data.zero_()
      net.features[0] = input_layer
  elif model_name == 'vgg19_orig':
    net = VGG.vgg19(pretrained=False, **kwargs)
    if in_channels != 3:
      input_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
      net.features[0] = input_layer
    init_weights_vgg_orig(net)
  elif model_name == 'alexnet':
    net = AlexNet(pretrained=pretrained, **kwargs)
    if in_channels != 3:
      input_layer = nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2)
      nn.init.kaiming_normal_(input_layer.weight, mode='fan_out', nonlinearity='relu')
      input_layer.bias.data.zero_()
      net.features[0] = input_layer
  elif model_name == 'lenet':
    kwargs['in_channels'] = in_channels
    net = lenet(**kwargs)
  else:
    raise ValueError("Unsupported model architecture.")
  return net
  
def init_weights_vgg_orig(net):
  """Initialize the weights of a CNN
     as described in https://arxiv.org/abs/1409.1556v1
  """
  for m in net.modules():
    if isinstance(m, nn.Conv2d):
      m.weight.data.normal_(0, 1e-2)
      if m.bias is not None:
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
      m.weight.data.normal_(0, 0.01)
      if m.bias is not None:
        m.bias.data.zero_()
  
def lenet(**kwargs):
  """Custom LeNet 9-Layer model
  """
  in_channels = kwargs.pop("in_channels", 3)
  model = LeNet(make_layers(cfg['lenet'], in_channels=in_channels), **kwargs)
  return model

class LeNet(nn.Module):
  def __init__(self, features, num_classes=10, init_weights=True):
    super(LeNet, self).__init__()
    self.features = features
    self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
    self.classifier = nn.Sequential(
      nn.Linear(3 * 3 * 64, 120),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(120, 84),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(84, num_classes),
    )
    if init_weights:
      self._initialize_weights()
      
  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
   
  def _initialize_weights(self):
    gain = nn.init.calculate_gain('relu')
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
          m.bias.data.zero_()

def make_layers(cfg, in_channels=3):
  layers = []
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)
  
cfg = {
       'lenet' : [6, 6, 'M', 16, 16, 'M', 64, 64, 'M']
      }
