# -*- coding: utf-8 -*-

"""Model definitions
"""

from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional
from torch import Tensor

__all__ = ['mlp', 'mlp_bn', 'wrn-40-2']

MLPOutputs = namedtuple('MLPOutputs', ['logits', 'h1', 'h2', 'h3', 'h4'])
MLPOutputs.__annotations__ = {'logits': torch.Tensor, 'h1': Optional[torch.Tensor], 'h2': Optional[torch.Tensor], 'h3': Optional[torch.Tensor], 'h4': Optional[torch.Tensor]}

# for backwards compatibility
_MLPOutputs = MLPOutputs

def load_model(model_name, classes=10, intermediate=False, **kwargs):
  """Load the specified architecture for CIFAR10
  
    Args:
      model_name: architecture type
      classes: number of predicted classes
  """
  kwargs['intermediate'] = intermediate
  
  if model_name == 'mlp':
    net = mlp(classes, input_dim=3072, width=4096, bottleneck_dim=1024, batchnorm=False, **kwargs)
  elif model_name == 'mlp_bn':
    net = mlp(classes, input_dim=3072, width=4096, bottleneck_dim=1024, batchnorm=True, **kwargs)
  elif model_name == 'wrn-40-2' or model_name == 'wideresnet':
    net = wideresnet(classes=10, depth=40, widen_factor=2, dropout=0.2)
  else:
    raise ValueError("Unsupported model architecture.")
  return net

def wide_resnet40_2(**kwargs):
  kwargs['width_per_group'] = 64*2
  return _resnet('wide_resnet40_2', BottleNeck, []) # FIXME find config

def mlp(num_classes, input_dim, width, bottleneck_dim, batchnorm, **kwargs):
  """MLP architecture with access to intermediate representations
     according to the paper Variational Information for Knowledge Distillation
     
  Args:
    intermediate (bool): If True, return intermediate representations from a forward pass
     
  """
  if batchnorm:
    kwargs = {'batchnorm' : True}
  return MLP(input_dim, num_classes, width, bottleneck_dim, **kwargs)
  

class BottleNeckLinear(nn.Module):

  def __init__(self, input_dim, bottleneck_dim, dropout=0.2, batchnorm=False, **kwargs):
    super(BottleNeckLinear, self).__init__()
    kwargs.pop('batchnorm', None)
    self.fc1 = nn.Linear(input_dim, bottleneck_dim, **kwargs)
    self.fc2 = nn.Linear(bottleneck_dim, input_dim, **kwargs)
    self.dropout = nn.Dropout(dropout)
    if batchnorm:
      self.bn = nn.BatchNorm1d(num_features=input_dim)
    else:
      self.bn = None
    self.relu = nn.ReLU(True)
  
  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.dropout(x)
    if self.bn is not None:
      x = self.bn(x)
    x = self.relu(x)
    return x

class BasicLinear(nn.Module):
  def __init__(self, input_dim, output_dim, dropout=0.2, batchnorm=False, **kwargs):
    super(BasicLinear, self).__init__()
    kwargs.pop('batchnorm', None)
    self.fc = nn.Linear(input_dim, output_dim, **kwargs)
    self.dropout = nn.Dropout(dropout)
    if batchnorm:
      self.bn = nn.BatchNorm1d(num_features=output_dim)
    else:
      self.bn = None
    self.relu = nn.ReLU(True)
  
  def forward(self, x):
    x = self.fc(x)
    x = self.dropout(x)
    if self.bn is not None:
      x = self.bn(x)
    x = self.relu(x)
    return x

class MLP(nn.Module):

  def __init__(self, input_dim, num_classes=10, width=4096, bottleneck=1024, intermediate=False, init_weights=True, **kwargs):
    super(MLP, self).__init__()
    kwargs.pop('intermediate', None)
    self.intermediate = intermediate
    self.hidden1 = BasicLinear(input_dim, width, **kwargs)
    self.hidden2 = BottleNeckLinear(width, bottleneck, **kwargs)
    self.hidden3 = BottleNeckLinear(width, bottleneck, **kwargs)
    self.hidden4 = BottleNeckLinear(width, bottleneck, **kwargs)
    self.classifier = nn.Linear(width, num_classes)
    
    if init_weights:
      for m in self.modules():
        if isinstance(m, nn.Linear):
          nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
          nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
    
  def _forward(self, x):
    x = torch.flatten(x, 1)
    if self.intermediate:
      h1 = self.hidden1(x)
      h2 = self.hidden2(h1)
      h3 = self.hidden3(h2)
      h4 = self.hidden4(h3)
      x = self.classifier(h4)
    else:
      x = self.hidden1(x)
      x = self.hidden2(x)
      x = self.hidden3(x)
      x = self.hidden4(x)
      x = self.classifier(x)
      h1, h2, h3, h4 = None, None, None, None
      
    return x, h1, h2, h3, h4
    
  @torch.jit.unused
  def eager_outputs(self, x, h1, h2, h3, h4):
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> MLPOutputs
    if self.training and self.intermediate:
      return MLPOutputs(x, h1, h2, h3, h4)
    else:
      return x
      
  def forward(self, x):
    x, h1, h2, h3, h4 = self._forward(x)
    if torch.jit.is_scripting():
      return MLPOutputs(x, h1, h2, h3, h4)
    else:
      return self.eager_outputs(x, h1, h2, h3, h4)
