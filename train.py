# -*- coding: utf-8 -*-

"""Model training
"""
import os
import sys
import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import data_loader
import models
import scores
import snapshot
import utils

def load_optimizer(args, net):
  """Load the optimizer specified by args.
  """ 
  scheduler = None
  if args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_step > 0 and args.lr_decay > 0:
      scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
  elif args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
  else:
    raise ValueError('Optimizer not supported.')
  return optimizer, scheduler

def get_writer(args, tb_logdir):
  """Get tensorboard summary writer
  """
  nclasses = args.subsample_classes if args.subsample_classes else data_loader.num_classes(args.dataset)
  
  writer = SummaryWriter(log_dir=tb_logdir)
  # log hyperparameters
  writer.add_text('Architecture', args.arch,0) # arch
  writer.add_text('Dataset', args.dataset,0) # dataset
  writer.add_text('Classes', str(nclasses), 0) # classes
  writer.add_text('Validation split size', str(args.split), 0) # split
  stratified = 'yes' if args.stratified else 'no'
  augmentation = 'yes' if args.augmentation else 'no'
  noise = 'pixel shuffle' if args.noise < 0. else str(args.noise)
  writer.add_text('Stratified sampling', stratified, 0) # stratified sampling
  writer.add_text('Data augmentation', augmentation, 0) # data augmentation
  writer.add_text('Noise', noise, 0) # data/label noise
  writer.add_text('Shuffle data at each epoch', str(args.shuffle), 0) # data shuffling
  writer.add_text('Batch size',str(args.batch_size),0) # batch size
  writer.add_text('Optimizer',args.optimizer,0) # optimizer
  writer.add_text('Momentum',str(args.momentum),0) # momentum
  writer.add_text('Weight decay',str(args.weight_decay),0) # weight decay
  writer.add_text('Max epochs', str(args.epochs), 0) # epochs
  writer.add_text('Start epoch', str(args.start_epoch), 0) # start epoch
  lr_policy = "lr rescaled by {} after {} epochs".format(args.lr_decay, args.lr_step)
  writer.add_text('Lr polcy',lr_policy,0) # lr policy
  pretrained = 'pretrained' if args.pretrained else 'scratch'
  writer.add_text('Pretrained', pretrained, 0) # pretrained model?
  if args.seed is not None:
    writer.add_text('Seed', str(args.seed), 0) # torch seed
  if args.split_seed is not None:
    writer.add_text('Validation split seed', str(args.split_seed), 0) # split seed
  if args.noise_seed is not None:
    writer.add_text('Noise seed', str(args.noise_seed), 0) # noise seed
  if args.class_sample_seed is not None:
    writer.add_text('Class subsample seed', str(args.class_sample_seed), 0) # class subsample seed
  return writer

def train(model, end_epoch, train_loader, optimizer, criterion, scheduler, device, snapshot_dirname, start_epoch=0, snapshot_every=0, val_loader=None, kill_plateaus=False, best_acc1=0, writer=None, snapshot_all_until=0, filename='net', train_acc=False):
  """Train the specified model according to user options.
  
    Args:
    
    model (nn.Module) -- the model to be trained
    end_epoch (int) -- maximum number of epochs
    train_loader (nn.DataLoader) -- train set loader
    optimizer (torch.optim optimizer) -- the optimizer to use
    criterion -- loss function to use
    scheduler -- learning rate scheduler
    device (torch.device) -- device to use
    start_epoch (int) -- starting epoch (useful for resuming training)
    snapshot_every (int) -- frequency of snapshots (in epochs)
    test_loader (optional, nn.DataLoader) -- test set loader
    train_acc (bool) -- whether to report accuracy on the train set
    
  """
  converged = True # used to kill models that plateau
  top1_prec = 0.
  if snapshot_every < 1:
    snapshot_every = end_epoch
  
  start_loss = 0.
  for epoch in range(start_epoch, end_epoch):
    # training loss
    avg_loss = 0.
    epoch_loss = 0.
    for batch_idx, (x, target) in enumerate(train_loader):
      optimizer.zero_grad()
      x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
      out = model(x)
      loss = criterion(out, target)
      avg_loss = avg_loss * 0.99 + loss.item() * 0.01
      epoch_loss += loss.item()
      loss.backward()
      optimizer.step()
      
      # report training loss
      if ((batch_idx+1) % 100 == 0) or ((batch_idx+1) == len(train_loader)):
        utils.print_train_loss(epoch, avg_loss, batch_idx, len(train_loader))
    # report training loss over epoch
    epoch_loss /= len(train_loader)
    utils.print_train_loss_epoch(epoch, epoch_loss)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    
    if scheduler is not None:
      scheduler.step()
      writer.add_scalar('Lr', scheduler.get_lr()[0], epoch)
      
    if (epoch < snapshot_all_until) or ((epoch +1) % snapshot_every == 0) or ((epoch +1) == end_epoch):
      top1_acc, top5_acc = 0, 0
      if val_loader is not None:
        val_loss, top1_acc, top5_acc = scores.evaluate(model, val_loader, criterion, device, topk=(1,5))
        utils.print_val_loss(epoch, val_loss, top1_acc, top5_acc)
        model = model.train()
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val/top1', top1_acc, epoch)
        writer.add_scalar('Accuracy/val/top5', top5_acc, epoch)
        
        # check whether training is stalling
        if kill_plateaus:
          if top1_prec == top1_acc:
            logger.debug("Prec val accuracy: {}, current val accuracy: {}. Model unlikely to converge. Quitting.".format(top1_prec, top1_acc))
            converged = False
            return model, converged
          else:
            top1_prec = top1_acc
      
      if train_acc:
        train_loss, top1_train, top5_train = scores.evaluate(model, train_loader, criterion, device, topk=(1,5))
        utils.print_train_loss_epoch(epoch, train_loss, top1_train, top5_train)
        model = model.train()
        writer.add_scalar('Accuracy/train/top1', top1_train, epoch)
        writer.add_scalar('Accuracy/train/top5', top5_train, epoch)
      # save snapshot
      snapshot.save_snapshot(model, optimizer, scheduler, epoch, top1_acc, top5_acc, filename, snapshot_dirname)
  return model, converged

def main(args):
  # set up project directories
  tb_logdir, snapshot_dir = prepare_dirs(args)
  # get logger
  logger = logging.getLogger('train')
  # tensorboard writer
  writer = get_writer(args, tb_logdir)
  
  use_cuda = torch.cuda.is_available() and args.cuda
  
  # set manual seed if required
  if args.seed is not None:
    torch.manual_seed(args.seed)
    if use_cuda:
      torch.cuda.manual_seed_all(args.seed)
  
  # check for cuda supports
  if use_cuda:
    device = torch.device("cuda:0")
    cudnn.benchmark = False # disabled to ensure reproducibility
  else:
    device = torch.device("cpu")
  
  # snapshot frequency
  if args.snapshot_every > 0 and not args.evaluate:
    logger.info('Saving snapshots to {}'.format(snapshot_dir))
  
  # load model
  classes, in_channels = data_loader.num_classes(args.dataset)
  if args.subsample_classes > 0:
    classes = args.subsample_classes
  net = models.load_model(args.arch, classes=classes, pretrained=args.pretrained, in_channels=in_channels)
  
  if args.pretrained and args.resume_from == '':
    logger.info('Loading pretrained {} on ImageNet.'.format(args.arch))
  else:
    logger.info('Creating model {}.'.format(args.arch))
  
  if torch.cuda.device_count() > 1:
    logger.info("Running on {} GPUs".format(torch.cuda.device_count()))
    net.features = torch.nn.DataParallel(net.features)
  
  # move net to device
  net = net.to(device=device)
  
  # get data loader for the specified dataset
  train_loader, test_loader, val_loader = data_loader.load_dataset(args.dataset, args.data_path, args.batch_size, shuffle=args.shuffle,
                                          augmentation=args.augmentation, noise=args.noise, split=args.split,
                                          num_workers=args.workers, split_seed=args.split_seed, noise_seed=args.noise_seed,
                                          stratified=args.stratified, nclasses=args.subsample_classes,
                                          class_sample_seed=args.class_sample_seed, no_normalization=args.unnormalize)
  # define loss
  criterion = nn.CrossEntropyLoss().to(device)
  
  start_epoch = args.start_epoch
  best_acc1, best_acc5 = 0, 0
  # load model from file
  if os.path.isfile(args.resume_from):
    # resume training given state dictionary
    optimizer, scheduler = load_optimizer(args, net)
    try:
      net, optimizer, scheduler, start_epoch, best_acc1, best_acc5 = snapshot.load_snapshot(net, optimizer,
                                                                     scheduler, args.resume_from, device)
    except KeyError:
      classes, in_channels = data_loader.num_classes(args.dataset)
      if args.subsample_classes > 0:
        classes = args.subsample_classes
      net = snapshot.load_model(args.arch, classes, args.resume_from, device, in_channels)
  else:
    # define optimizer
    optimizer, scheduler = load_optimizer(args, net)
  
  # evaluate model
  if args.evaluate:
    val_loss, top1_acc, top5_acc = scores.evaluate(net, test_loader, criterion, device, topk=(1,5))
    utils.print_val_loss(args.epochs, val_loss, top1_acc, top5_acc)
    writer.add_scalar('Loss/test', val_loss, args.epochs)
    writer.add_scalar('Accuracy/test/top1', top1_acc, args.epochs)
    writer.add_scalar('Accuracy/test/top5', top5_acc, args.epochs)
    writer.close()
    return
    
  if args.evaluate_train:
    train_loss, top1_acc, top5_acc = scores.evaluate(net, train_loader, criterion, device, topk=(1,5))
    utils.print_train_loss_epoch(args.epochs, train_loss, top1_acc, top5_acc)
    if best_acc1 * best_acc5 > 0:
      # if nonzero, print best val accuracy
      utils.print_val_loss(args.epochs, -1., best_acc1, best_acc5)
    writer.add_scalar('Loss/train', train_loss, args.epochs)
    writer.add_scalar('Accuracy/train/top1', top1_acc, args.epochs)
    writer.add_scalar('Accuracy/train/top5', top5_acc, args.epochs)
    writer.close()
    return
  
  utils.print_model_config(args)
  
  if start_epoch == 0:
    pretrained = 'pretrained_' if args.pretrained else 'init_'
    filename = args.arch + '_' + pretrained + str(start_epoch) + '.pt'
    logger.info("Saving model initialization to {}".format(filename))
    snapshot.save_model(net, filename, snapshot_dir)
  
  # train the model
  net.train()
  if val_loader is None and test_loader is not None:
    val_loader = test_loader
    logger.warning("Using TEST set to validate model during training!")
  net, converged = train(net, args.epochs, train_loader, optimizer, criterion, scheduler, 
                    device, snapshot_dirname=snapshot_dir, start_epoch=start_epoch, 
                    snapshot_every=args.snapshot_every, val_loader=val_loader, 
                    kill_plateaus=args.kill_plateaus, best_acc1=best_acc1, 
                    writer=writer, snapshot_all_until=args.snapshot_all_until, filename=args.arch, train_acc=args.train_acc)
  if test_loader is not None:
    val_loss, top1_acc, top5_acc = scores.evaluate(net, test_loader, criterion, device, topk=(1,5))
    utils.print_val_loss(args.epochs, val_loss, top1_acc, top5_acc)
    net = net.train()
    writer.add_scalar('Loss/test', val_loss, args.epochs)
    writer.add_scalar('Accuracy/test/top1', top1_acc, args.epochs)
    writer.add_scalar('Accuracy/test/top5', top5_acc, args.epochs)

  # save final model
  if converged:
    pretrained = 'pretrained_' if args.pretrained else ''
    filename = args.arch + '_' +  pretrained + str(args.epochs) + '.pt'
    snapshot.save_model(net, filename, snapshot_dir)

  writer.close()

def get_arg_parser():
  """Parse command line arguments and return parser
  """
  # parse command line arguments
  import argparse
  parser = argparse.ArgumentParser(description="Model training/finetuning.")

  # models
  parser.add_argument("--arch", type=str, default=None, help="Network architecture to be trained. Run without this option to see a list of all supported archs.")
  # dataset
  parser.add_argument("--dataset", type=str, default=None, help="Dataset to train the network on. Run without this option to see a list of supported datasets.")
  # subsample classes
  parser.add_argument("--subsample-classes", type=int, default=0, help="Subsample only SUBSAMPLE_CLASSES classes from DATASET. If set to 0 (default) all classes of DATASET are used.")
  # subsample seed
  parser.add_argument("--class-sample-seed", type=int, default=None, help="Numpy random seed used for sampling classes.")
  # ratio of noisy labels
  parser.add_argument("--noise", type=float, default=0., help="Ratio of corrupted labels, in [0., 1.]. Set to -1 to enable pixel shuffle in place of label noise.")
  # noise seed
  parser.add_argument("--noise-seed", type=int, default=None, help="Numpy seed for corrupting labels.")
  # data augmentation
  parser.add_argument("--augmentation", action='store_true', default=False, help="Enable data augmentation.")
  # pretrained model from pytorch zoo
  parser.add_argument("--pretrained", action='store_true', default=False, help="Load pretrained architecture on ImageNet from model zoo.")
  # validation split
  parser.add_argument("--split", type=float, default=0., help="Validation split size. The resutling split is class-unbalanced [default=0].")
  # stratified sampling
  parser.add_argument("--stratified", action='store_true', default=False, help="Validation split with equal balance of samples per class.")
  # numpy split seed
  parser.add_argument("--split-seed", type=int, default=None, help="Seed used for shuffling the data before making the validation split.")
  # shuffle training data before splitting
  parser.add_argument("--shuffle", action='store_true', default=False, help="Shuffle the training data before making the validation split [default=False].")
  # evalute model only
  parser.add_argument("--evaluate", action='store_true', default=False, help="Evaluate model on the validation set.")
  # evalute model on train set
  parser.add_argument("--evaluate-train", action='store_true', default=False, help="Evaluate model on the training set.")
  # cuda support
  parser.add_argument("--cuda", action='store_true', default=False, help="Enable GPU support.")
  # number of parallel jobs
  parser.add_argument("--workers", "-j", type=int, default=1, help="Number of parallel data processing jobs.")
  # datasets
  parser.add_argument("--data-path", type=str, default='./data', help="Path to local ImageNet folder.")
  # number of training epochs
  parser.add_argument("--epochs", type=int, default=20, help="The number of epochs used for training [default = 20].")
  # start epoch
  parser.add_argument("--start-epoch", type=int, default=0, help="Starting epoch for training.")
  # minibatch size
  parser.add_argument("--batch-size", type=int, default=128, help="The minibatch size for training [default = 128].")
  # the optimizer used to train each base classifier
  parser.add_argument("--optimizer", type=str, default='sgd', help="Supported optimizers: sgd, adam [default = sgd].")
  # base learning rate for SGD training
  parser.add_argument("--lr", type=float, default=0.001, help="The base learning rate for SGD optimization [default = 0.001].")
  # sgd step size
  parser.add_argument("--lr-step", type=int, default=0, help="The step size (# iterations) of the learning rate decay [default = off].")
  # learning rate decay factor
  parser.add_argument("--lr-decay", type=float, default=0., help="The decay factor of the learning rate decay [default = off].")
  # weight decay
  parser.add_argument("--weight-decay", type=float, default=1e-4, help="The weight decay coefficient [default = 1e-4].")
  # momentum for sgd
  parser.add_argument("--momentum", type=float, default=0.9, help="The momentum coefficient for SGD [default = 0.9].")
  # path where to store/load models
  parser.add_argument("--models-path", type=str, default='./models', help="The dirname where to store/load models [default = './models'].")
  # pytorch seed
  parser.add_argument("--seed", type=int, default=None, help="Pytorch PRNG seed.")
  # tensorboard directory
  parser.add_argument("--tb-logdir", type=str, default='./tensorboard', help="The tensorboard log folder [default = './tensorboard'].")
  # snapshot frequency
  parser.add_argument("--snapshot-every", type=int, default=0, help="Snapshot the model state every E epochs [default = 0].")
  # snapshot every epoch until the specified one, then snapshot according to --snapshot-every
  parser.add_argument("--snapshot-all-until", type=int, default=0, help="Optional. Snapshot every epoch until the specified one, then snapshot according to the --snapshot-every argument.")
  # path to a model snapshot, used to continue training
  parser.add_argument("--resume-from", type=str, default='', help="Path to a model snapshot [default = None].")
  parser.add_argument("--kill-plateaus", default=False, action='store_true', help="Quit training if the model validation accuracy plateaus in the first 10 epochs.")
  parser.add_argument("--train-acc", default=False, action='store_true', help="(Optional) compute train accuracy and report it during training.")
  parser.add_argument("--unnormalize", default=False, action='store_true', help="Disable data normalization and represent instead pixel values in [0,1]. [Default = False].")
  # log file
  parser.add_argument("--log", type=str, default='train.log', help="Logfile name [default = 'train.log'].")

  # parse the command line
  args = parser.parse_args()
  return args
  
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


def prepare_dirs(args):
  """Prepare all the project dirs, creating those which don't exist.
  
    Return:
      tb_logdir (str): tensorboard log directory
      snapshot_dir (str): model snapshot directory
  """
  from datetime import datetime
  
  if args.subsample_classes > 0:
    dataset = args.dataset + '_' + str(args.subsample_classes)
  else:
    dataset = args.dataset
  
  # add dataset to paths
  snapshot_dir = os.path.join(args.models_path, dataset)
  log_dir = os.path.join('./log', dataset)
  tb_logdir = os.path.join(args.tb_logdir, dataset)
  
  # noisy labels
  if args.noise < 0.:
    snapshot_dir = os.path.join(snapshot_dir, 'pixel_shuffle')
    log_dir = os.path.join(log_dir, 'pixel_shuffle')
    tb_logdir = os.path.join(tb_logdir, 'pixel_shuffle')
  elif args.noise > 0.:
    noise = 'noise_' + str(int(args.noise * 100.))
    snapshot_dir = os.path.join(snapshot_dir, noise)
    log_dir = os.path.join(log_dir, noise)
    tb_logdir = os.path.join(tb_logdir, noise)
  
  # add arch to paths
  snapshot_dir = os.path.join(snapshot_dir, args.arch)
  log_dir = os.path.join(log_dir, args.arch)
  tb_logdir = os.path.join(tb_logdir, args.arch)
  
  # add seed to paths
  if args.seed is not None:
    snapshot_dir = os.path.join(snapshot_dir, str(args.seed))
    log_dir = os.path.join(log_dir, str(args.seed))
    tb_logdir = os.path.join(tb_logdir, str(args.seed))
    
  # add pruning config to tb_logdir
  if os.path.exists(args.resume_from):
    splits = args.resume_from.split('_')
    if 'pruned' in splits:
      conf = os.path.splitext(splits[2])[0]
      tb_logdir = os.path.join(tb_logdir, str(conf))
  
  if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if not os.path.exists(tb_logdir):
    os.makedirs(tb_logdir)
    
  # prepend date to logfile for sorting
  date = datetime.now().strftime('%Y%m%d-%H%M%S_')
  logfile = os.path.join(log_dir, date + args.log)
  
  logger = init_logger('train', logfile)
  logger.info('Using {} as model snapshot directory.'.format(snapshot_dir))
  
  return tb_logdir, snapshot_dir

if __name__ == "__main__":
  import signal
  signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

  args = get_arg_parser()
  
  if args.arch is None:
    print("Supported architectures:")
    print(models.__all__)
    sys.exit(0)
    
  if args.dataset is None:
    print("Supported datasets:")
    print(data_loader.__all__)
    sys.exit(0)
    
  main(args)
