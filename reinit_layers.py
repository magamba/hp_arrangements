# -*- coding: utf-8 -*-

"""Reproduce main experiment from
   "Are all layers created equal?"
"""
import os
import sys
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import data_loader
import models
import snapshot
import utils
import scores

def plot_heatmaps(results, filename, plots_path=None):
  """Plot heatmap of loss in accuracy
  """
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    print("Package matplotlib required to plot heatmaps. Aborting.")
    sys.exit(1)
  
  try:
    import seaborn as sns
  except ImportError:
    print("Package seaborn required to plot heatmaps. Aborting.")
    sys.exit(1)
  
  # set matplotlib fontsize
  plt.rcParams.update({'font.size' : 20}) # was 24
  plt.rcParams.update({'axes.titlesize' : 22}) # was 26
  plt.rcParams.update({'axes.labelsize' : 20}) # was 24
  plt.rcParams.update({'lines.linewidth' : 4})
  plt.rcParams.update({'lines.markersize' : 20})
  plt.rcParams.update({'xtick.labelsize' : 20}) # was 24
  plt.rcParams.update({'ytick.labelsize' : 20}) # was 24
  
  best_top1_acc = results["top1_test"]
  nepochs = len(results["init_from_epoch"])
  nlayers = len(results["init_from_epoch"][next(iter(results["init_from_epoch"]))])
  
  xlabels = range(1, nlayers+1)
  ylabels = list(results["init_from_epoch"].keys())
  
  acc_loss = np.zeros((nepochs, nlayers), dtype=np.float)
  best_acc = np.ones(nlayers, dtype=np.float) * float(best_top1_acc)
  
  for epoch_id, epoch in enumerate(results["init_from_epoch"]):
    accuracies = np.array(results["init_from_epoch"][epoch])
    acc_loss[epoch_id] = (best_acc - accuracies) / best_top1_acc
  
  ax = sns.heatmap(acc_loss, xticklabels=xlabels, yticklabels=ylabels, linewidth=0.5)
  
  if plots_path is not None:
    filename = os.path.join(plots_path, filename + '.pdf')
    plt.savefig(filename, bbox_inches="tight", dpi=300)
  else:
    plt.show()

def reinit_weights(results, net, net_init, test_loader, criterion, epoch, device):
  """For each convolutional layer in net, reinitialize its weights
     with the values of the corresponding weights in net_init.
     
     Store the test accuracy of the resulting model in results.
     
     results = {
                 "epoch_id": [acc1, ..., a_l, ..., acc_L],
               } where a_l is the test accuracy of net with layer l replaced by the
                 corresponding layer in net_init.
  """
  logger = logging.getLogger('reinit_weights')
  if results is None:
    results = { "init_from_epoch" : {} }
    
  test_accs = []
  net.eval()
  net_init.eval()
  
  block_id, stage_id = 0, 0
  conv_counter = 0
  init_type = "random initialization." if str(epoch) == "rand" else "weights of epoch {}.".format(epoch)
  for layer_id, (layer, layer_init) in enumerate(zip(net.features, net_init.features)):
    if isinstance(layer, nn.Conv2d):
      if layer.kernel_size == (1,1):
        continue
      
      conv_counter +=1
      # replace layer
      logger.info("Reinitializing parameters of layer {} from {}".format(conv_counter, init_type))  
      weight_copy = layer.weight.clone()
      if layer.bias is not None:
        bias_copy = layer.bias.clone()
        layer.bias.data.copy_(layer_init.bias.data)
      layer.weight.data.copy_(layer_init.weight.data)
      
      _, top1_acc, _ = scores.evaluate(net, test_loader, criterion, device, topk=(1,5))
      test_accs.append(top1_acc)
      
      # restore original parameter
      layer.weight.data.copy_(weight_copy.data)
      if layer.bias is not None:
        layer.bias.data.copy_(bias_copy.data)
            
      block_id += 1
    elif isinstance(layer, nn.MaxPool2d):
      stage_id += 1
      block_id = 0
  try:
    _ = results["init_from_epoch"]
  except KeyError:
    results["init_from_epoch"] = {}
    
  results["init_from_epoch"][str(epoch)] = test_accs
  return results
  

def init_filenames(file_handler):
  """Yield each line of filename
  """
  for line in file_handler:
    line = line.strip()
    yield line

def write_metadata(results, filename, args, top1_acc, top5_acc):
  """Write network metadata to results dictionary
  """
  results["arch"] = args.arch
  results["dataset"] = args.dataset
  results["top1_test"] = top1_acc
  results["top5_test"] = top5_acc
  if args.seed is not None:
    results["seed"] = str(args.seed)
  splits = os.path.splitext(filename)[0].split('_')
  if 'init' in splits:
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
  parser = argparse.ArgumentParser(description="Reinit network weights and compute drop in performance.")

  # architecture
  parser.add_argument("--arch", type=str, default=None, help="Network architecture. Run without this option to see a list of all available pretrained archs.")
  # dataset
  parser.add_argument("--dataset", type=str, default=None, help="Dataset used to train the model. Used to specify the number of classes of the prediction layer of the model.")
  # upscale image data
  parser.add_argument("--upscale", action='store_true', default=False, help="Upscale image data to 244x224 pixels.")
  # upscale and pad image data
  parser.add_argument("--upscale-padding", action='store_true', default=False, help="Upscale image data to 112x112 pixels and then zero-pad to 224x224.")
  # subsample classes
  parser.add_argument("--subsample-classes", type=int, default=0, help="Subsample only SUBSAMPLE_CLASSES classes from DATASET. If set to 0 (default) all classes of DATASET are used.")
  # subsample seed
  parser.add_argument("--class-sample-seed", type=int, default=None, help="Numpy random seed used for sampling classes.")
  # load trained model from file
  parser.add_argument("--load-from", type=str, default='', help="Load trained network from file.")
  # measure distance from initialization
  parser.add_argument("--inits-from", type=str, default='', help="Network initialization. Specify a model snapshot to measure distance from initialization for each layer.")
  # cuda support
  parser.add_argument("--cuda", action='store_true', default=False, help="Enable GPU support.")
  # results path
  parser.add_argument("--results", type=str, default='./results', help="Path to store results [default = './results'].")
  # plots path
  parser.add_argument("--plots", type=str, default='./plots', help="Path to store plots [default = './plots'].")
  # log file
  parser.add_argument("--log", type=str, default='reinit_weights.log', help="Logfile name [default = 'reinit_weights.log'].")
  # seed
  parser.add_argument("--seed", type=int, default=None, help="Pytorch seed. (Optional)If using --load-from, specify the seed the model was trained with.")
  # number of parallel jobs
  parser.add_argument("--workers", "-j", type=int, default=1, help="Number of parallel data processing jobs.")
  # datasets
  parser.add_argument("--data-path", type=str, default='./data', help="Path to local dataset folder.")
  # minibatch size
  parser.add_argument("--batch-size", type=int, default=128, help="The minibatch size for training [default = 128].")
  # randomize weights
  parser.add_argument("--rand", default=False, action='store_true', help="Perform weight randomization test.")

  # parse the command line
  args = parser.parse_args()
  return args

def prepare_dirs(args):
  """Prepare directories to store results
  """
  dataset = args.dataset
  if args.subsample_classes > 0:
    dataset = dataset + '_' + str(args.subsample_classes)
  results_path = os.path.join(args.results, dataset)
  results_path = os.path.join(results_path, args.arch)
  
  plots_path = os.path.join(args.plots, dataset)
  plots_path = os.path.join(plots_path, args.arch)
  
  logs_path = os.path.join('./log', dataset)
  logs_path = os.path.join(logs_path, args.arch)
  
  if args.seed is not None:
    results_path = os.path.join(results_path, str(args.seed))
    plots_path = os.path.join(plots_path, str(args.seed))
    logs_path = os.path.join(logs_path, str(args.seed))
  
  if not os.path.exists(results_path):
    os.makedirs(results_path)
    
  if not os.path.exists(plots_path):
    os.makedirs(plots_path)
  
  if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
  logfile = os.path.join(logs_path, args.log)
    
  logger = init_logger('reinit_weights', logfile)
  
  return results_path, plots_path

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
    
  results_path, plots_path = prepare_dirs(args)
  logger = logging.getLogger('reinit_weights')
    
  if torch.cuda.is_available() and args.cuda:
    device = torch.device("cuda:0")
    cudnn.benchmark = True
  else:
    device = torch.device("cpu")
  
  classes, in_channels = data_loader.num_classes(args.dataset)
  if args.subsample_classes > 0:
    classes = args.subsample_classes
 
  if os.path.exists(args.load_from):
    logger.info("Loading {} from {}.".format(args.arch, args.load_from))
    net = snapshot.load_model(args.arch, classes, args.load_from, device, in_channels)
  else:
    logger.info("Cannot load trained model from {}: no such file.".format(args.load_from))
    sys.exit(1)
      
  net = net.to(device)
  criterion = nn.CrossEntropyLoss().to(device)
  
  # load test set
  _, test_loader, _ = data_loader.load_dataset(args.dataset, args.data_path, args.batch_size, shuffle=True,
                                  augmentation=False, num_workers=args.workers, nclasses=args.subsample_classes,
                                  class_sample_seed=args.class_sample_seed, upscale=args.upscale, upscale_padding=args.upscale_padding)
  # evaluate model
  logger.info("Evaluating trained model on test set.")
  test_loss, top1_acc, top5_acc = scores.evaluate(net, test_loader, criterion, device, topk=(1,5))
  utils.print_val_loss(0, test_loss, top1_acc, top5_acc, 'reinit_weights')
  
  results={}
  if os.path.exists(args.inits_from):
    logger.info("Loading network weights initializations from {}.".format(args.inits_from))
    # get generator
    with open(args.inits_from, 'r') as fp:
      for init_file in init_filenames(fp):
        if os.path.exists(init_file):
          logger.info("Loading network weights from {}".format(init_file))
          net_init = snapshot.load_model(args.arch, classes, init_file, device, in_channels)
        else:
          logger.warning("Warning. File not found: {}. Skipping.".format(init_file))
          continue
      
        splits = os.path.splitext(init_file)[0].split('_')
        if 'init' in splits:
          epoch = 0
        else:
          epoch = int(splits[-1]) +1
        results = reinit_weights(results, net, net_init, test_loader, criterion, epoch, device)
  
  if args.rand:
    # load random initialization
    logger.info("Loading random initialization.")
    random_init = models.load_model(args.arch, classes, pretrained=False, in_channels=in_channels)
    # randomize weights and compute accuracy
    results = reinit_weights(results, net, random_init, test_loader, criterion, "rand", device)
  
  if os.path.exists(args.load_from):
    filename = os.path.splitext(os.path.basename(args.load_from))[0]
    filename = 'reinit_' + filename
  
  results = write_metadata(results, args.load_from, args, top1_acc, top5_acc)
  
  # save results to json
  logger.info("Saving results to file.")
  snapshot.save_results(results, filename, results_path)
  
  # plot results
  logger.info("Plotting results.")
  plot_heatmaps(results, filename, plots_path)
  
if __name__ == "__main__":
  args = get_arg_parser()
  main(args)
