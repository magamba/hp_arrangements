# -*- coding: utf-8 -*-

import logging
import os

def print_json(json_dict):
  """Pretty prints a json dictionary
  """
  import json
  parsed = json.loads(json.dumps(json_dict).encode("utf-8"))
  print(json.dumps(parsed, indent=2, sort_keys=True))

def print_model_config(args, logger_name='train'):
  """ Print the model configuration defined by the user.
  
  Args
    args -- arg parser containing user options
  """
  logger = logging.getLogger(logger_name)
  
  model_config_str = "\nModel configuration: \n"+ \
                   "\t seed: {}\n".format(args.seed or 'random') + \
                   "\t architecture: {}\n".format(args.arch) + \
                   "\t dataset: {}\n".format(args.dataset) + \
                   "\t start epoch: {}\n".format(args.start_epoch) + \
                   "\t epochs: {}\n".format(args.epochs) + \
                   "\t batch size: {}\n".format(args.batch_size) + \
                   "\t base lr: {:.5f}\n".format(args.lr) + \
                   "\t optimizer: {}\n".format(args.optimizer)
  if args.optimizer == 'sgd':
    model_config_str = model_config_str + \
                   "\t momentum: {}\n".format(args.momentum) + \
                   "\t weight decay: {}\n".format(args.weight_decay) + \
                   "\t lr rescaled by {} after {} steps\n".format(args.lr_decay, args.lr_step)
  model_config_str = model_config_str +  \
                     "\t saving snapshots every {} epochs to {}\n".format(args.snapshot_every, args.models_path)
  logger.info(model_config_str)
  
def print_val_loss(epoch, avg_loss, top1_acc, top5_acc, logger_name='train'):
  """ Print the average validation loss and accuracy
  """
  logger = logging.getLogger(logger_name)
  logger.info('\t epoch: {}, test loss: {:.6f}, top_1 acc: {:.3f}, top_5 acc:{:.3f}'.format(
        epoch, avg_loss, top1_acc, top5_acc))
        
def print_train_loss(epoch, avg_loss, batch_idx, num_batches, logger_name='train'):
  """Print the running average of the train loss for the given batch
  """
  logger = logging.getLogger(logger_name)
  logger.info('\t epoch: {}, batch: {}/{}, train loss: {:.6f}'.format(
          epoch, batch_idx+1, num_batches, avg_loss))
          
def print_train_loss_epoch(epoch, epoch_loss, top1_acc=None, top5_acc=None, logger_name='train'):
  """Print the average train loss for the specified epoch
  """
  logger = logging.getLogger(logger_name)
  msg = '\t epoch: {}, train loss: {:.6f}'.format(epoch, epoch_loss)
  if top1_acc is not None:
    msg += ', top_1 acc: {:.3f}'.format(top1_acc)
  if top5_acc is not None:
    msg += ', top_5 acc: {:.3f}'.format(top5_acc)
  logger.info(msg)
  
def print_regularization_loss_epoch(epoch, regularization_loss, logger_name='train'):
  """Print the regularization loss for the specified epoch
  """
  logger = logging.getLogger(logger_name)
  logger.info('\t epoch: {}, regularization loss: {:.6f}'.format(epoch, regularization_loss))

