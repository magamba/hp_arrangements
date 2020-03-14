# -*- coding: utf-8 -*-

import pytest
import numpy as np
import data_loader

class TestNoisyLabels:

  def test_mnist_noisy_0(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum()
    assert diff == 0.
    

  def test_mnist_noisy_20(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.2)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.19

  def test_mnist_noisy_40(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.4)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.39

  def test_mnist_noisy_60(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False)
 
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.6)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.59
    

  def test_mnist_noisy_80(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.8)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.79

  def test_mnist_noisy_100(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=1.0)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.98
    
  def test_mnist_noisy_10(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.1)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.09
    
  def test_cifar10_noisy_0(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False, noise=0.)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum()
    assert diff == 0.
    

  def test_cifar10_noisy_20(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False, noise=0.2)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.19

  def test_cifar10_noisy_40(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False, noise=0.4)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.39

  def test_cifar10_noisy_60(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False)
 
    train_noisy, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False, noise=0.6)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.59
    

  def test_cifar10_noisy_80(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False, noise=0.8)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.79

  def test_cifar10_noisy_100(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False, noise=1.0)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.98
    
  def test_cifar10_noisy_10(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False)
  
    train_noisy, _, _ = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, shuffle=False, noise=0.1)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / len(clean_labels)
    assert diff > 0.09
    
  def test_mnist_noisy_0_split_20(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, split_seed=42, split=0.2)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0., split_seed=42, split=0.2)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum()
    assert diff == 0.
    

  def test_mnist_noisy_20_split_20(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, split_seed=42, split=0.2)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.2, split_seed=42, split=0.2)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / (len(clean_labels) * 0.8)
    assert diff > 0.19

  def test_mnist_noisy_40_split_20(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, split_seed=42, split=0.2)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.4, split_seed=42, split=0.2)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / (len(clean_labels) * 0.8)
    assert diff > 0.39

  def test_mnist_noisy_60_split_20(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, split_seed=42, split=0.2)
 
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.6, split_seed=42, split=0.2)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / (len(clean_labels) * 0.8)
    assert diff > 0.59
    

  def test_mnist_noisy_80_split_20(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, split_seed=42, split=0.2)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.8, split_seed=42, split=0.2)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / (len(clean_labels) * 0.8)
    assert diff > 0.79

  def test_mnist_noisy_100_split_20(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, split_seed=42, split=0.2)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=1.0, split_seed=42, split=0.2)
    
    noisy_labels = train_noisy.dataset.targets
    
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / (len(clean_labels) * 0.8)
    assert diff > 0.98
    
  def test_mnist_noisy_10_split_20(self):
    batch_size = 128
    train_set, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, split_seed=42, split=0.2)
  
    train_noisy, _, _ = data_loader.load_dataset('mnist', './data', batch_size=batch_size, shuffle=False, noise=0.1, split_seed=42, split=0.2)
    
    noisy_labels = train_noisy.dataset.targets
    clean_labels = train_set.dataset.targets
    
    diff = (np.array(clean_labels) != np.array(noisy_labels)).sum() / (len(clean_labels) * 0.8)
    assert diff > 0.09

if __name__ == '__main__':
  import sys
  from os import path
  sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
  
