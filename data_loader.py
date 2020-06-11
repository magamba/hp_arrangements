# -*- coding: utf-8 -*-

""" Load ImageNet train/test/validation
"""
import sys
import os
import torch
import numpy as np
import logging

import torch.utils.data
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

__all__ = ['cifar10', 'cifar100', 'imagenet', 'mnist']

def load_dataset(dataset_name, dataset_path, batch_size, shuffle=True, augmentation=True, noise=0., split=0., num_workers=4, pin_memory=True, split_seed=None, noise_seed=None, stratified=False, nclasses=0, class_sample_seed=42, no_normalization=False, upscale=False, upscale_padding=False):
  """Load the specified dataset
  """
  if dataset_name == 'imagenet':
    if noise != 0.:
      raise NotImplementedError("Noisy labels / pixels on ImageNet are currently not supported.")
    train_loader, test_loader, val_loader = load_imagenet(dataset_path, batch_size, shuffle=shuffle, augmentation=augmentation,
                                            split=split, num_workers=num_workers, pin_memory=pin_memory,
                                            split_seed=split_seed, stratified=stratified, nclasses=nclasses, 
                                            sample_seed=class_sample_seed, no_normalization=no_normalization)
  elif dataset_name == 'cifar10':
    train_loader, test_loader, val_loader = load_cifar10(dataset_path, batch_size, shuffle=shuffle, augmentation=augmentation,
                                            noise=noise, split=split, num_workers=num_workers, pin_memory=pin_memory,
                                            split_seed=split_seed, noise_seed=noise_seed, stratified=stratified,
                                            no_normalization=no_normalization, upscale=upscale, upscale_padding=upscale_padding)
  elif dataset_name == 'cifar100':
    train_loader, test_loader, val_loader = load_cifar100(dataset_path, batch_size, shuffle=shuffle, augmentation=augmentation,
                                            noise=noise, split=split, num_workers=num_workers, pin_memory=pin_memory,
                                            split_seed=split_seed, noise_seed=noise_seed, stratified=stratified,
                                            no_normalization=no_normalization, upscale=upscale, upscale_padding=upscale_padding)
  elif dataset_name == 'mnist':
    train_loader, test_loader, val_loader = load_mnist(dataset_path, batch_size, shuffle=shuffle, augmentation=augmentation,
                                            noise=noise, split=split, num_workers=num_workers, pin_memory=pin_memory,
                                            split_seed=split_seed, noise_seed=noise_seed, stratified=stratified,
                                            no_normalization=no_normalization)
  else:
    raise ValueError("Unsupported dataset.")
  return train_loader, test_loader, val_loader
  
def num_classes(dataset):
  """Return the number of classes of dataset
  """
  if dataset == 'cifar10':
    classes = 10
    in_channels = 3
  elif dataset == 'cifar100':
    classes = 100
    in_channels = 3
  elif dataset == 'imagenet':
    classes = 1000
    in_channels = 3
  elif dataset == 'mnist':
    classes = 10
    in_channels = 1
  else:
    raise ValueError("Unsupported dataset.")
  return classes, in_channels
  
######################
###### DATASETS ######
######################

def load_cifar10(dataset_path, batch_size, shuffle=True, augmentation=True, noise=0., split=0., num_workers=4, pin_memory=True, split_seed=None, noise_seed=None, stratified=False, no_normalization=False, upscale=False, upscale_padding=False):
  """Load CIFAR10
  """
  if upscale:
    train_transforms = [transforms.Resize((224,224))]
    test_transforms = [transforms.Resize((224,224)),
                       transforms.ToTensor()]
    crop_size = 224
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.1953, 0.1925, 0.1942]
  elif upscale_padding:
    train_transforms = [transforms.Resize((112,112)),
                        transforms.Pad(56)]
    test_transforms = [transforms.Resize((112,112)),
                       transforms.Pad(56),
                       transforms.ToTensor()]
    crop_size = 224
    mean = [0.1229, 0.1206, 0.1117]
    std = [0.2367, 0.2323, 0.2190]
  else:
    train_transforms = []
    test_transforms = [transforms.ToTensor()]
    crop_size = 32
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
  
  if augmentation:
    train_transforms += [transforms.RandomCrop(crop_size, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()]
  else:
    train_transforms += [transforms.ToTensor()]
  
  normalize = transforms.Normalize(mean = mean,
                                    std = std)
  
  if not no_normalization:
    train_transforms.append(normalize)
    test_transforms.append(normalize)
    
  train_transform = transforms.Compose(train_transforms)
  test_transform = transforms.Compose(test_transforms)
                 
  train_indices, val_indices = None, None
  if split > 0.:
    validation_set = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=test_transform)
    dataset_size = len(validation_set)
    train_indices, val_indices = split_dataset(dataset_size, split, shuffle, split_seed, stratified, validation_set.targets)
  
  if noise != 0.:
    if split > 0.:
      train_set = CIFAR10NoisyLabels(noise_ratio=noise, noise_seed=noise_seed, train_indices=train_indices,
                  root=dataset_path, train=True, transform=train_transform, download=False)
    else:
      train_set = CIFAR10NoisyLabels(noise_ratio=noise, noise_seed=noise_seed, root=dataset_path, train=True, 
                  transform=train_transform, download=False)
  else:
    train_set = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
  test_set = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=test_transform)
  
  if train_indices is not None:
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)
  else:
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=num_workers, pin_memory=pin_memory)
  if val_indices is not None:
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, sampler=val_sampler,
                                             num_workers=num_workers, pin_memory=pin_memory)
  else:
    val_loader = None

  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=pin_memory)
  return train_loader, test_loader, val_loader

def load_cifar100(dataset_path, batch_size, shuffle=True, augmentation=True, noise=0., split=0., num_workers=4, pin_memory=True, split_seed=None, noise_seed=None, stratified=False, no_normalization=False, upscale=False, upscale_padding=False):
  """Load CIFAR100
  """
  if upscale:
    train_transforms = [transforms.Resize((224,224))]
    test_transforms = [transforms.Resize((224,224)),
                       transforms.ToTensor()]
    crop_size = 224
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.1942, 0.1918, 0.1958]
  elif upscale_padding:
    train_transforms = [transforms.Resize((112,112)),
                        transforms.Pad(56)]
    test_transforms = [transforms.Resize((112,112)),
                       transforms.Pad(56),
                       transforms.ToTensor()]
    crop_size = 224
    mean = [0.1268, 0.1217, 0.1103]
    std = [0.2435, 0.2343, 0.2177]
  else:
    train_transforms = []
    test_transforms = [transforms.ToTensor()]
    crop_size = 32
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2009, 0.1984, 0.2023]
  
  if augmentation:
    train_transforms += [transforms.RandomCrop(crop_size, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()]
  else:
    train_transforms += [transforms.ToTensor()]

  normalize = transforms.Normalize(mean = mean,
                                    std = std)

  if not no_normalization:
    train_transforms.append(normalize)
    test_transforms.append(normalize)
    
  train_transform = transforms.Compose(train_transforms)
  test_transform = transforms.Compose(test_transforms)
  
  train_indices, val_indices = None, None
  if split > 0.:
    validation_set = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=test_transform)
    dataset_size = len(validation_set)
    train_indices, val_indices = split_dataset(dataset_size, split, shuffle, split_seed, stratified, validation_set.targets)
  
  if noise != 0.:
    if split > 0.:
      train_set = CIFAR100NoisyLabels(noise_ratio=noise, noise_seed=noise_seed, train_indices=train_indices,
                  root=dataset_path, train=True, transform=train_transform, download=False)
    else:
      train_set = CIFAR100NoisyLabels(noise_ratio=noise, noise_seed=noise_seed, root=dataset_path, train=True, 
                  transform=train_transform, download=False)
  else:
    train_set = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=train_transform)
  test_set = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=test_transform)
  
  if train_indices is not None:
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)
  else:
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=num_workers, pin_memory=pin_memory)
  if val_indices is not None:
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, sampler=val_sampler,
                                             num_workers=num_workers, pin_memory=pin_memory)
  else:
    val_loader = None

  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=pin_memory)
  
  return train_loader, test_loader, val_loader
  
def load_mnist(dataset_path, batch_size, shuffle=True, augmentation=True, noise=0., split=0., num_workers=4, pin_memory=True, split_seed=None, noise_seed=None, stratified=False, no_normalization=False):
  """Load MNIST
  """
  normalize = transforms.Normalize(mean = [0.1307], std = [0.3015])
  
  if augmentation:
    train_transforms = [transforms.RandomCrop(28, padding=4),
                       transforms.ToTensor()]
  else:
    train_transforms = [transforms.ToTensor()]
  
  test_transforms = [transforms.ToTensor()]
                 
  if not no_normalization:
    train_transforms.append(normalize)
    test_transforms.append(normalize)
    
  train_transform = transforms.Compose(train_transforms)
  test_transform = transforms.Compose(test_transforms)
  
  train_indices, val_indices = None, None
  if split > 0.:
    validation_set = datasets.MNIST(root=dataset_path, train=True, download=True, transform=test_transform)
    dataset_size = len(validation_set)
    train_indices, val_indices = split_dataset(dataset_size, split, shuffle, split_seed, stratified, validation_set.targets)
  
  if noise != 0.:
    if split > 0.:
      train_set = MNISTNoisyLabels(noise_ratio=noise, noise_seed=noise_seed, train_indices=train_indices,
                  root=dataset_path, train=True, transform=train_transform, download=False)
    else:
      train_set = MNISTNoisyLabels(noise_ratio=noise, noise_seed=noise_seed, root=dataset_path, train=True, 
                  transform=train_transform, download=False)
  else:
    train_set = datasets.MNIST(root=dataset_path, train=True, download=True, transform=train_transform)
  test_set = datasets.MNIST(root=dataset_path, train=False, download=True, transform=test_transform)
  
  if train_indices is not None:
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)
  else:
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=num_workers, pin_memory=pin_memory)
  if val_indices is not None:
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, sampler=val_sampler,
                                             num_workers=num_workers, pin_memory=pin_memory)
  else:
    val_loader = None

  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=pin_memory)
  
  return train_loader, test_loader, val_loader

def load_imagenet(dataset_path, batch_size, shuffle=True, augmentation=True, split=0., num_workers=4, pin_memory=True, split_seed=None, stratified=False, nclasses=1000, sample_seed=42, no_normalization=False):
  """Load ImageNet train and test set from the specified path.
  """
  if not os.path.exists(dataset_path):
    raise ValueError("Please specify a valid path to your local ImageNet folder.")
    
  traindir = os.path.join(dataset_path, 'train')
  testdir = os.path.join(dataset_path, 'val')

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
  
  if augmentation:
    train_transforms = [transforms.RandomResizedCrop(224),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()]
  else:
    train_transforms = [transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()]
  
  val_transforms = [transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()]
                 
  if not no_normalization:
    train_transforms.append(normalize)
    val_transforms.append(normalize)
    
  train_transforms = transforms.Compose(train_transforms)
  val_transforms = transforms.Compose(val_transforms)
                                    
  train_dataset = datasets.ImageFolder(traindir, train_transforms)
  val_dataset = datasets.ImageFolder(traindir, val_transforms)
  test_dataset = datasets.ImageFolder(testdir, val_transforms)
  
  if nclasses > 0 and nclasses < len(train_dataset.classes):
    subsample_classes(train_dataset, nclasses, shuffle=shuffle, sample_seed=sample_seed)
    subsample_classes(val_dataset, nclasses, shuffle=shuffle, sample_seed=sample_seed)
    subsample_classes(test_dataset, nclasses, shuffle=shuffle, sample_seed=sample_seed)
  idx_to_label = train_dataset.targets
  
  train_indices, val_indices = None, None
  if split > 0.:
    train_size = len(train_dataset)
    train_indices, val_indices = split_dataset(train_size, split, shuffle, split_seed, stratified, idx_to_label)
  
  if train_indices is not None:
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)
  else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=num_workers, pin_memory=pin_memory)
  if val_indices is not None:
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                             num_workers=num_workers, pin_memory=pin_memory)
  else:
    val_loader = None

  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                             num_workers=num_workers, pin_memory=pin_memory)
  return train_loader, test_loader, val_loader
  
#######################
##### SUBSAMPLING #####
#######################

def subsample_classes(dataset, nclasses, shuffle=True, sample_seed=42):
  """Sample nclasses random classes from dataset
  
    Arguments:
      dataset (torchvision.datasets.ImageFolder) dataset to subsample from
      nclasses (int) number of classes to keep
      shuffle (bool) whether to shuffle classes before subsampling
      sample_seed (int) numpy random seed used for shuffling
  
    Return
      subsampled dataset (modified in-place)
  """
  classes = dataset.classes
  if shuffle:
    if sample_seed is not None:
      np.random.seed(sample_seed)
    np.random.shuffle(classes)
  subset = classes[:nclasses]
  subset_idx = [dataset.class_to_idx[c] for c in subset]
  
  samples = []
  targets = []
  class_to_idx = {}
  for s in dataset.samples:
    for index, ss in enumerate(subset_idx):
      if s[1] == ss:
        samples.append((s[0], index))
        targets.append(index)
        class_to_idx[classes[index]] = index
        break
  
  dataset.samples = samples
  dataset.classes = subset
  dataset.targets = targets
  dataset.class_to_idx = class_to_idx

def split_dataset(train_size, split_size, shuffle=True, split_seed=42, stratified=False, idx_to_label=None):
  """Generate train and validation split given the length of a dataset
  
     If stratified is True, a list idx_to_label should be provided,
     so that idx_to_label[i] is the label of the i-th sample in the train set
  """
  
  if split_size <= 0. or split_size >= 1.:
    raise ValueError("The size ratio between validation and training set should be in (0,1).")
  
  if stratified:
    try:
      from sklearn.model_selection import train_test_split
    except ImportError:
      print("Package sklearn is required for stratified sampling.")
      sys.exit(0)
    
    indices = np.arange(train_size)
    labels = np.unique(idx_to_label)
    train_indices, val_indices, _, _ = train_test_split(indices, idx_to_label, 
                                       test_size=split_size, random_state=split_seed, shuffle=shuffle, stratify=labels)
  else:
    indices = list(range(train_size))
    
    if shuffle:
      if split_seed is not None:
        np.random.seed(split_seed)
      np.random.shuffle(indices)
      
    split_index = train_size - int(np.floor(train_size * split_size))
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
  
  return train_indices, val_indices
  
######################
### NOISY DATASETS ###
######################

class CIFAR10NoisyLabels(datasets.CIFAR10):
  """Load CIFAR10 with noisy labels
  
  improved from: https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
  
  Args:
    noise (float) -- ratio of corrupted labels, in [0, 1]
    noise_seed (int) - seed for label corruption
    train_indices (list) - indices of samples to corrupt (optional). Used for data splits.    
  """
  
  def __init__(self, noise_ratio=0., noise_seed=12345, train_indices=None, classes=10, **kwargs):
    super(CIFAR10NoisyLabels, self).__init__(**kwargs)
    self.classes = classes
    self.noise_ratio = noise_ratio
    if noise_ratio > 0.:
      self.corrupt_labels(noise_ratio, noise_seed, train_indices)
    elif noise_ratio < 0.:
      self.shuffle_pixels(noise_seed, train_indices)
      
  def __getitem__(self, index):
    img, target = super(CIFAR10NoisyLabels, self).__getitem__(index)
    if self.noise_ratio < 0.:
      perm = self.perms[index]
      if perm is not None:
        idx_permute = torch.from_numpy(perm)
        transform = torchvision.transforms.Lambda(lambda x: x.view(3, -1)[:, idx_permute].view(3,32,32))
        img = transform(img)
    return img, target
  
  def shuffle_pixels(self, noise_seed, train_indices):
    from numpy.random import RandomState
    if noise_seed is not None:
      self.prng = RandomState(noise_seed)
    else:
      self.prng = RandomState()
    
    try:
      labels = np.array(self.targets)
    except AttributeError: # older torchvision version
      if self.train:
        labels = np.array(self.train_labels)
      else:
        labels = np.array(self.test_labels)
    
    if train_indices is not None:
      split_mask = np.zeros(len(labels), dtype=bool)
      for idx in train_indices:
        split_mask[idx] = True
    else:
      split_mask = np.ones(len(labels), dtype=bool)
      
    # make list of permutations
    perms = []
    for idx in range(len(split_mask)):
      if split_mask[idx]:
        perms.append(self.prng.permutation(1024))
      else:
        perms.append(None)
    self.perms = perms
    
  def corrupt_labels(self, noise_ratio, noise_seed, train_indices):
    logger = logging.getLogger('train')
    logger.info('Randomizing CIFAR10 labels')
    
    try:
      labels = np.array(self.targets)
    except AttributeError: # older torchvision version
      if self.train:
        labels = np.array(self.train_labels)
      else:
        labels = np.array(self.test_labels)
    
    np.random.seed(noise_seed) # fixed seed for reproducibility
    
    # mask for training indices
    if train_indices is not None:
      split_mask = np.zeros(len(labels), dtype=bool)
      for idx in train_indices:
        split_mask[idx] = True
    else:
      split_mask = np.ones(len(labels), dtype=bool)
    
    train_labels = labels[split_mask]
    orig_labels = train_labels.copy() # sanity check
    
    noise_mask = np.random.rand(len(train_labels)) <= noise_ratio
    random_labels = np.random.choice(range(1, self.classes), noise_mask.sum())
    train_labels[noise_mask] = (train_labels[noise_mask] + random_labels) % self.classes
    
    labels[split_mask] = train_labels
    labels = [int(x) for x in labels]
    
    logger.debug('Sanity check -- actual ratio of corrupted labels: {}%'.format(np.sum(orig_labels != np.array(train_labels)) / len(train_labels)))
    
    try:
      self.targets = labels
    except AttributeError: # older torchvision version
      if self.train:
        self.train_labels = labels
      else:
        self.test_labels  = labels
        
class CIFAR100NoisyLabels(datasets.CIFAR100):
  """Load CIFAR100 with noisy labels
  
  improved form https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
  
  Args:
    noise (float) -- ratio of corrupted labels, in [0, 1]
    noise_seed (int) - seed for label corruption
    train_indices (list) - indices of samples to corrupt (optional). Used for data splits.    
  """
  
  def __init__(self, noise_ratio=0., noise_seed=12345, train_indices=None, classes=100, **kwargs):
    super(CIFAR100NoisyLabels, self).__init__(**kwargs)
    self.classes = classes
    self.noise_ratio = noise_ratio
    if noise_ratio > 0.:
      self.corrupt_labels(noise_ratio, noise_seed, train_indices)
    elif noise_ratio < 0.:
      self.shuffle_pixels(noise_seed, train_indices)
      
  def __getitem__(self, index):
    img, target = super(CIFAR100NoisyLabels, self).__getitem__(index)
    if self.noise_ratio < 0.:
      perm = self.perms[index]
      if perm is not None:
        idx_permute = torch.from_numpy(perm)
        transform = torchvision.transforms.Lambda(lambda x: x.view(3, -1)[:, idx_permute].view(3,32,32))
        img = transform(img)
    return img, target
  
  def shuffle_pixels(self, noise_seed, train_indices):
    from numpy.random import RandomState
    if noise_seed is not None:
      self.prng = RandomState(noise_seed)
    else:
      self.prng = RandomState()
    
    try:
      labels = np.array(self.targets)
    except AttributeError: # older torchvision version
      if self.train:
        labels = np.array(self.train_labels)
      else:
        labels = np.array(self.test_labels)
    
    if train_indices is not None:
      split_mask = np.zeros(len(labels), dtype=bool)
      for idx in train_indices:
        split_mask[idx] = True
    else:
      split_mask = np.ones(len(labels), dtype=bool)
      
    # make list of permutations
    perms = []
    for idx in range(len(split_mask)):
      if split_mask[idx]:
        perms.append(self.prng.permutation(1024))
      else:
        perms.append(None)
    self.perms = perms
    
  def corrupt_labels(self, noise_ratio, noise_seed, train_indices):
    logger = logging.getLogger('train')
    logger.info('Randomizing CIFAR100 labels')
    
    try:
      labels = np.array(self.targets)
    except AttributeError: # older torchvision version
      if self.train:
        labels = np.array(self.train_labels)
      else:
        labels = np.array(self.test_labels)
    
    np.random.seed(noise_seed) # fixed seed for reproducibility
    
    # mask for training indices
    if train_indices is not None:
      split_mask = np.zeros(len(labels), dtype=bool)
      for idx in train_indices:
        split_mask[idx] = True
    else:
      split_mask = np.ones(len(labels), dtype=bool)
    
    train_labels = labels[split_mask]
    orig_labels = train_labels.copy() # sanity check
    
    noise_mask = np.random.rand(len(train_labels)) <= noise_ratio
    random_labels = np.random.choice(range(1, self.classes), noise_mask.sum())
    train_labels[noise_mask] = (train_labels[noise_mask] + random_labels) % self.classes
    
    labels[split_mask] = train_labels
    labels = [int(x) for x in labels]
    
    logger.debug('Sanity check -- actual ratio of corrupted labels: {}%'.format(np.sum(orig_labels != np.array(train_labels)) / len(train_labels)))
    
    try:
      self.targets = labels
    except AttributeError: # older torchvision version
      if self.train:
        self.train_labels = labels
      else:
        self.test_labels  = labels

class MNISTNoisyLabels(datasets.MNIST):
  """Load MNIST with noisy labels
  
  improved from: https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
  
  Args:
    noise (float) -- ratio of corrupted labels, in [0, 1]
    noise_seed (int) - seed for label corruption
    train_indices (list) - indices of samples to corrupt (optional). Used for data splits.    
  """
  
  def __init__(self, noise_ratio=0., noise_seed=12345, train_indices=None, classes=10, **kwargs):
    super(MNISTNoisyLabels, self).__init__(**kwargs)
    self.classes = classes
    self.noise_ratio = noise_ratio
    if noise_ratio > 0.:
      self.corrupt_labels(noise_ratio, noise_seed, train_indices)
    elif noise_ratio < 0.:
      self.shuffle_pixels(noise_seed, train_indices)
      
  def __getitem__(self, index):
    img, target = super(MNISTNoisyLabels, self).__getitem__(index)
    if self.noise_ratio < 0.:
      perm = self.perms[index]
      if perm is not None:
        idx_permute = torch.from_numpy(perm)
        transform = torchvision.transforms.Lambda(lambda x: x.view(1, -1)[:, idx_permute].view(1,32,32))
        img = transform(img)
    return img, target
  
  def shuffle_pixels(self, noise_seed, train_indices):
    from numpy.random import RandomState
    if noise_seed is not None:
      self.prng = RandomState(noise_seed)
    else:
      self.prng = RandomState()
    
    try:
      labels = np.array(self.targets)
    except AttributeError: # older torchvision version
      if self.train:
        labels = np.array(self.train_labels)
      else:
        labels = np.array(self.test_labels)
    
    if train_indices is not None:
      split_mask = np.zeros(len(labels), dtype=bool)
      for idx in train_indices:
        split_mask[idx] = True
    else:
      split_mask = np.ones(len(labels), dtype=bool)
      
    # make list of permutations
    perms = []
    for idx in range(len(split_mask)):
      if split_mask[idx]:
        perms.append(self.prng.permutation(1024))
      else:
        perms.append(None)
    self.perms = perms
    
  def corrupt_labels(self, noise_ratio, noise_seed, train_indices):
    logger = logging.getLogger('train')
    logger.info('Randomizing MNIST labels')
    
    try:
      labels = np.array(self.targets)
    except AttributeError: # older torchvision version
      if self.train:
        labels = np.array(self.train_labels)
      else:
        labels = np.array(self.test_labels)
    
    np.random.seed(noise_seed) # fixed seed for reproducibility
    
    # mask for training indices
    if train_indices is not None:
      split_mask = np.zeros(len(labels), dtype=bool)
      for idx in train_indices:
        split_mask[idx] = True
    else:
      split_mask = np.ones(len(labels), dtype=bool)
    
    train_labels = labels[split_mask]
    orig_labels = train_labels.copy() # sanity check
    
    noise_mask = np.random.rand(len(train_labels)) <= noise_ratio
    random_labels = np.random.choice(range(1, self.classes), noise_mask.sum())
    train_labels[noise_mask] = (train_labels[noise_mask] + random_labels) % self.classes
    
    labels[split_mask] = train_labels
    labels = [int(x) for x in labels]
    
    logger.debug('Sanity check -- actual ratio of corrupted labels: {}%'.format(np.sum(orig_labels != np.array(train_labels)) / len(train_labels)))
    
    try:
      self.targets = labels
    except AttributeError: # older torchvision version
      if self.train:
        self.train_labels = labels
      else:
        self.test_labels  = labels

