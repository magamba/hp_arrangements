# -*- coding: utf-8 -*-

"""Compute projection statistics for the specified
   convolutional network.
"""
import os
import sys
import json
import numpy as np
import logging

import torch
import torch.nn as nn

import data_loader
import models
import snapshot

def kernel_all_positive_proj(conv_w, normalize=False):
  """Given a tensor representing the weights of a convolutional layer,
     compute the projection of the vector (1, 1, ..., 1) with the
     (unnormalized) normal vector to the first hyperplane 
     induced by the convolutional kernel, identified by the first row
     of the vectorized convolutional weight tensor.
     
     Args:
      conv_w: torch tensor of shape (N,C,k,k)
      
     Return
      proj: list of projections
  """  
  conv_w = conv_w.view(conv_w.size(0), -1) # reshape W as NxCkk
  shape = conv_w.shape
  proj = conv_w.sum(1).cpu().numpy() # compute projection
  if normalize:
    ones = torch.ones(shape[1])
    norm_ones = torch.sqrt(ones.sum(0)).item()
    w_norm = torch.sqrt((conv_w.clone() ** 2).sum(1)).cpu().numpy()
    proj /= (w_norm * norm_ones)
  proj = np.squeeze(proj).tolist()
  return proj
  
def distance_from_init(net, net_init, proj_dict=None):
  """Compute distance from initialization for each layer of net.
     Store measures to proj_dict if specified, or return new dictionary otherwise.
  """
  if proj_dict is None:
    proj_dict = {}
  
  block_id, stage_id = 0, 0
  for layer, layer_init in zip(net.features, net_init.features):
    if isinstance(layer, nn.Conv2d):
      if layer.kernel_size == (1,1):
        continue
      try:
        proj_dict[str(stage_id)]
      except KeyError:
        proj_dict[str(stage_id)] = {}
        
      param = list(layer.parameters())[0]
      param_init = list(layer_init.parameters())[0]
      l2_dist, inf_dist = distance_from_init_layer(param.detach(), param_init.detach())
      proj_dict[str(stage_id)][str(block_id)]["l2_dist"] = l2_dist
      proj_dict[str(stage_id)][str(block_id)]["l_inf_dist"] = inf_dist
      block_id += 1
    elif isinstance(layer, nn.MaxPool2d):
      stage_id += 1
      block_id = 0
  
  return proj_dict 
  
def distance_from_init_layer(param, param_init):
  """Compute the L2 and L_infinity distance from initialization of 
     param from param_init.
  """
  assert param.shape == param_init.shape, "Shape mismatch. param: {}, param_init: {}".format(param.size, param_init.size)
  
  init_inf_norm = param_init.norm(p=float('inf'))
  init_l2_norm = param_init.norm(p='fro')
  
  l2_dist = (param - param_init).norm(p='fro') / init_l2_norm
  inf_dist = (param - param_init).norm(p=float('inf')) / init_inf_norm
  
  l2_dist = l2_dist.cpu().numpy().item()
  inf_dist = inf_dist.cpu().numpy().item()
  
  return l2_dist, inf_dist
  
def probability_mass(observations, k=0):
  """Compute P(X <= k) for a set of observations from a scalar random variable X.
  """
  try:
    from statsmodels.distributions.empirical_distribution import ECDF
  except ImportError:
    print("FATAL: missing package statsmodels Please install it to make use of --cdf.")
    sys.exit(0)
  cdf = ECDF(observations)
  mean = np.mean(observations)
  mass = cdf(k)
  return mean, mass

def num_conv_layers(net):
  """Compute the number of convolutional layers in net
  """
  params = list(net.features.parameters())
  conv_layer_count = 0
  for layer in params:
    if isinstance(layer, nn.Conv2d):
      conv_layer_count += 1 
  return conv_layer_count  

def positive_orthant(net, normalize=False):
  """For each layer in a convolutional network, compute the 
     all positive projection of (1, 1, ..., 1) w.r.t the hyperplanes
     induced by each convolutional kernel.
     
     Return:
       dict = {
        "0" = { ... }, # stage 1
        "1" = {   # stage 2
          "0" = { # block 1
                  "projections" : [], # projection values
                  "mean" : mean, # mean of the observed distribution
                  "mass" : mass, # prob mass of negative projections
                }
          "1" = { ... } # block 2
          },
        "net" = {
                  "projections": [], # netwise projections
                  "mean" : mean, # netwise mean of observed distribution
                  "mass" : mass, # netwise prob mass of negative projections
                }
      }
  """
  proj_dict = {}
  proj_net = [] # projections for the entire network
  
  stage_id = 0
  block_id = 0
  for layer in net.features:
    if isinstance(layer, nn.Conv2d):
      if layer.kernel_size == (1,1):
        continue
      try:
        proj_dict[str(stage_id)]
      except KeyError:
        proj_dict[str(stage_id)] = {}
        
      param = list(layer.parameters())[0]
      kernel_proj = kernel_all_positive_proj(param.detach(), normalize)
      mean, mass = probability_mass(kernel_proj, k=0)
      proj_dict[str(stage_id)][str(block_id)] = { "projections" : kernel_proj, "mass" : mass, "mean" : mean }
      proj_net += kernel_proj
      block_id += 1
  
    elif isinstance(layer, nn.MaxPool2d):
      stage_id += 1
      block_id = 0
  
  mean_net, mass_net = probability_mass(proj_net, k=0)
  proj_dict["net"] = { "projections" : proj_net, "mass" : mass_net, "mean" : mean_net }
  return proj_dict

def perc_negative(param):
  """Return the percentage of negative projections for the
     convolutional layer represented by param
  """
  param = param.detach()
  param = param.view(param.size(0),-1) 
  perc = float((param.sum(1) <= 0).sum().item()) / param.size(0) * 100.
  return perc
  
def compute_perc(net):
  """Return the percentage of negative projections for each convolutional
     layer in network
  """
  perc_neg = []
  for layer in net.features:
    if isinstance(layer, nn.Conv2d):
      if layer.kernel_size == (1,1):
        continue
      param = list(layer.parameters())[0]
      perc_neg.append(perc_negative(param))
  return perc_neg
  
def write_metadata(results, filename, args):
  """Write network metadata to results dictionary
  """
  results["arch"] = args.arch
  results["dataset"] = args.dataset
  if args.seed is not None:
    results["seed"] = str(args.seed)
  splits = filename.split('_')
  if 'pretrained' in splits:
    epoch = -1
  elif 'init' in splits:
    epoch = 0
  else:
    epoch = int(splits[-1]) +1
  results["epoch"] = str(epoch)
  return results

def get_arg_parser():
  """Parse command line arguments and return parser
  """
  # parse command line arguments
  import argparse
  parser = argparse.ArgumentParser(description="Compute projection statistics.")

  # architecture
  parser.add_argument("--arch", type=str, default=None, help="Network architecture to be trained. Run without this option to see a list of all available pretrained archs.")
  # dataset
  parser.add_argument("--dataset", type=str, default=None, help="Dataset used to train the model. Used to specify the number of classes of the prediction layer of the model.")
  # load trained model from file
  parser.add_argument("--load-from", type=str, default='', help="Load trained network from file.")
  # measure distance from initialization
  parser.add_argument("--init-from", type=str, default='', help="(Optional) network initialization. Specify a model snapshot to measure distance from initialization for each layer.")
  # load pretrained models from model zoo
  parser.add_argument("--pretrained", action='store_true', default=False, help="Load pretrained architecture on ImageNet [default=False].")
  # cuda support
  parser.add_argument("--cuda", action='store_true', default=False, help="Enable GPU support.")
  
  # normalize projections
  parser.add_argument("--normalize", action='store_true', default=False, help="Normalize projections.")
  # results path
  parser.add_argument("--results", type=str, default='./results', help="Path to store results [default = './results'].")
  # log file
  parser.add_argument("--log", type=str, default='positive_orthant.log', help="Logfile name [default = 'positive_orthant.log'].")
  # seed
  parser.add_argument("--seed", type=int, default=None, help="Pytorch seed. (Optional)If using --load-from, specify the seed the model was trained with.")

  # parse the command line
  args = parser.parse_args()
  return args

def prepare_dirs(args):
  """Prepare directories to store results
  """
  results_path = os.path.join(args.results, args.dataset)
  results_path = os.path.join(results_path, args.arch)
  
  logs_path = os.path.join('./log', args.dataset)
  logs_path = os.path.join(logs_path, args.arch)
  
  if args.seed is not None:
    results_path = os.path.join(results_path, str(args.seed))
    logs_path = os.path.join(logs_path, str(args.seed))
  
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  
  if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
  logfile = os.path.join(logs_path, args.log)
    
  logger = init_logger('projections', logfile)
  
  return results_path

def init_logger(logger_name, logfile):
  """Init logger and sets logging options
  
    Args:
      logger_name  name of the logger instance
  """
  # init logging
  logger = logging.getLogger(logger_name)
  logger.setLevel(logging.DEBUG)
  f_handler = logging.FileHandler(logfile)
  f_handler.setLevel(logging.DEBUG)
  c_handler = logging.StreamHandler()
  c_handler.setLevel(logging.ERROR)
  
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  f_handler.setFormatter(formatter)
  c_handler.setFormatter(formatter)
  logger.addHandler(f_handler)
  logger.addHandler(c_handler)
  return logger

def main(args):
  """
    Main
  """
  if args.arch is None:
    print("Available architectures:")
    print(models.__all__)
    sys.exit(0)
    
  if args.dataset is None:
    print("Available datasets:")
    print(data_loader.__all__)
    sys.exit(0)
    
  # set manual seed if required
  if args.seed is not None:
    torch.manual_seed(args.seed)
    
  results_path = prepare_dirs(args)
  logger = logging.getLogger('projections')
    
  if torch.cuda.is_available() and args.cuda:
    device = torch.device("cuda:0")
  else:
    device = torch.device("cpu")
  
  classes, in_channels = data_loader.num_classes(args.dataset)
  if os.path.exists(args.load_from):
    logger.info("Loading {} from {}.".format(args.arch, args.load_from))
    net = snapshot.load_model(args.arch, classes, args.load_from, device, in_channels=in_channels)
    if args.pretrained:
      logger.warning("Warning: --pretrained should only be used when loading a pretrained from the model zoo. Ignoring.")
  else:
    try:
      net = models.load_model(args.arch, classes, pretrained=args.pretrained, in_channels=in_channels)
    except ValueError:
      print("Unsupported architecture: {}".format(args.arch))
      print("Supported architectures:")
      print(models.__all__)
      sys.exit(0)
    
    if args.pretrained:
      logger.info("Loading pretrained {} from torchvision model zoo".format(args.arch))
    else:
      logger.info("Could not find snapshot. Initializing network weights from scratch.")
      
  net = net.to(device)
  
  net_init = None
  if os.path.exists(args.init_from):
    logger.info("Loading network weights at initialization from {}. They will be used to compute the network's distance from initialization.".format(args.init_from))
    net_init = snapshot.load_model(args.arch, classes, args.init_from, device, in_channels)
  
  # compute statistics per network
  logger.info("Computing projections statistics...")
  results = positive_orthant(net, args.normalize)
  if net_init is not None:
    logger.info("Computing distance from initialization...")
    results = distance_from_init(net, net_init, results)
  logger.info("done")
  
  if os.path.exists(args.load_from):
    filename = os.path.splitext(os.path.basename(args.load_from))[0]
  elif args.pretrained:
    filename = 'pretrained_' + args.arch
  else:
    filename = 'init_' + args.arch
  
  results = write_metadata(results, filename, args)
  
  # save results to json
  logger.info("Saving results to file.")
  snapshot.save_results(results, filename, results_path)

if __name__ == "__main__":
  args = get_arg_parser()
  main(args)
