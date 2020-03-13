# -*- coding: utf-8 -*-

import os
import logging
import json

import torch
import torch.nn as nn
import models

"""Utilities to save and load models
"""

def save_results(results, filename, results_path):
  """Save a dictionary of results to json file
  """
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  filename = os.path.join(results_path, filename) + '.json'

  with open(filename, 'wb') as fp:
    fp.write(json.dumps(results).encode("utf-8"))

def save_model(net, filename, dirname):
  """Save a model indexed to the specified path.
  """
  logger = logging.getLogger('train')
  
  path = os.path.join(os.path.normpath(dirname), filename)
  logger.info('Saving model to {}'.format(path))
  if isinstance(net, torch.nn.DataParallel):
    torch.save(net.module.state_dict(), path)
  else:
    torch.save(net.state_dict(), path)

def load_model(model_name, classes, path, device, in_channels=3):
  """Load a model from file for inference.
  
  Keyword arguments:
  model_name (str) -- name of the model architecture
  classes -- number of predicted classes
  path (str) -- path to the saved model
  device (torch.device) -- where to move the model after loading
  """
  
  logger = logging.getLogger('train')
  
  net = models.load_model(model_name, classes=classes, pretrained=False, in_channels=in_channels)

  # load parameters
  logger.info('Loading model {} from {}'.format(model_name, path))
  checkpoint = torch.load(path, map_location=device)
  
  try:
    if isinstance(net, nn.DataParallel):
      net.module.load_state_dict(checkpoint['model_state_dict'])
    else:
      net.load_state_dict(checkpoint['model_state_dict'])
  except KeyError:
    net.load_state_dict(checkpoint)

  # move to device
  net = net.to(device = device)

  # set model to inference mode
  net = net.eval()

  return net

def save_snapshot(net, optimizer, scheduler, epoch, best_acc1, best_acc5, model_name, dirname):
  """Save snapshot of training
  """
  logger = logging.getLogger('train')
  
  filename = model_name + '_' + str(epoch) + '.tar'
  path = os.path.join(dirname, filename)
  
  model_state_dict = {}
  if isinstance(net, torch.nn.DataParallel):
    model_state_dict = net.module.state_dict()
  else:
    model_state_dict = net.state_dict()
   
  state_dictionary = {
            'epoch': epoch,
            'top1_acc': best_acc1,
            'top5_acc': best_acc5,
            'model_state_dict' : model_state_dict,
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : {}
            }
  if scheduler is not None:
    state_dictionary['scheduler_state_dict'] = scheduler.state_dict()
    
  logger.info('Saving snapshot for epoch {} to {}'.format(epoch, path))
  torch.save(state_dictionary, path)


def load_snapshot(net, optimizer, scheduler, filename, device):
  """Load a stored snapshot
  
  net (nn.Module) -- an model instance
  optimizer -- optimizer instance
  scheduler -- scheduler instance
  filename (str) -- path to the stored model (.tar)
  device (torch.device) -- device where to load the model to
  
  """
  logger = logging.getLogger('train')
  logger.info('Loading snapshot from {}'.format(filename))
  
  checkpoint = torch.load(filename, map_location=device)
  if isinstance(net, torch.nn.DataParallel):
    net.module.load_state_dict(checkpoint['model_state_dict'])
  else:
    net.load_state_dict(checkpoint['model_state_dict'])
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.zero_grad()
  if scheduler is not None:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  epoch = checkpoint['epoch']

  try:
    best_acc1 = checkpoint['top1_acc']
    best_acc5 = checkpoint['top5_acc']
  except KeyError:
    best_acc1, best_acc5 = 0., 0.
  
  net = net.train()
  return net, optimizer, scheduler, epoch, best_acc1, best_acc5

