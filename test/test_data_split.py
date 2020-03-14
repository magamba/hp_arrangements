# -*- coding: utf-8 -*-

import pytest
import numpy as np
import data_loader

class TestDataSplit:

  def test_mnist_split_0(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.)
    assert val_loader is None
    assert len(train_loader) == 600
    
  def test_mnist_shuffle_split_0(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.)
    assert val_loader is None
    assert len(train_loader) == 600
    
  def test_mnist_split_10(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.1)
    assert len(val_loader) == 60
    assert len(train_loader) == 540
    
  def test_mnist_shuffle_split_10(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.1)
    assert len(val_loader) == 60
    assert len(train_loader) == 540
    
  def test_mnist_split_20(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.2)
    assert len(val_loader) == 120
    assert len(train_loader) == 480
    
  def test_mnist_shuffle_split_20(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.2)
    assert len(val_loader) == 120
    assert len(train_loader) == 480
    
  def test_mnist_split_40(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.4)
    assert len(val_loader) == 240
    assert len(train_loader) == 360
    
  def test_mnist_shuffle_split_40(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.4)
    assert len(val_loader) == 240
    assert len(train_loader) == 360 
    
  def test_mnist_split_60(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.6)
    assert len(val_loader) == 360
    assert len(train_loader) == 240
    
  def test_mnist_shuffle_split_60(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.6)
    assert len(val_loader) == 360
    assert len(train_loader) == 240
    
  def test_mnist_split_80(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.8)
    assert len(val_loader) == 480
    assert len(train_loader) == 120
    
  def test_mnist_shuffle_split_80(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.8)
    assert len(val_loader) == 480
    assert len(train_loader) == 120
    
  def test_mnist_split_99(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.99)
    assert len(val_loader) == 594
    assert len(train_loader) == 6
      
  def test_mnist_shuffle_split_99(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('mnist', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.99)
    assert len(val_loader) == 594
    assert len(train_loader) == 6

  def test_cifar10_split_0(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.)
    assert val_loader is None
    assert len(train_loader) == 500
    
  def test_cifar10_shuffle_split_0(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.)
    assert val_loader is None
    assert len(train_loader) == 500
    
  def test_cifar10_split_10(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.1)
    assert len(val_loader) == 50
    assert len(train_loader) == 450
    
  def test_cifar10_shuffle_split_10(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.1)
    assert len(val_loader) == 50
    assert len(train_loader) == 450
    
  def test_cifar10_split_20(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.2)
    assert len(val_loader) == 100
    assert len(train_loader) == 400
    
  def test_cifar10_shuffle_split_20(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.2)
    assert len(val_loader) == 100
    assert len(train_loader) == 400
    
  def test_cifar10_split_40(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.4)
    assert len(val_loader) == 200
    assert len(train_loader) == 300
    
  def test_cifar10_shuffle_split_40(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.4)
    assert len(val_loader) == 200
    assert len(train_loader) == 300 
    
  def test_cifar10_split_60(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.6)
    assert len(val_loader) == 300
    assert len(train_loader) == 200
    
  def test_cifar10_shuffle_split_60(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.6)
    assert len(val_loader) == 300
    assert len(train_loader) == 200
    
  def test_cifar10_split_80(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.8)
    assert len(val_loader) == 400
    assert len(train_loader) == 100
    
  def test_cifar10_shuffle_split_80(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.8)
    assert len(val_loader) == 400
    assert len(train_loader) == 100
    
  def test_cifar10_split_99(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=False, augmentation=False, noise=0., split_seed=42, split=0.99)
    assert len(val_loader) == 495
    assert len(train_loader) == 5
      
  def test_cifar10_shuffle_split_99(self):
    batch_size = 100
    train_loader, test_loader, val_loader = data_loader.load_dataset('cifar10', './data', batch_size=batch_size, 
                                             shuffle=True, augmentation=False, noise=0., split_seed=42, split=0.99)
    assert len(val_loader) == 495
    assert len(train_loader) == 5

if __name__ == '__main__':
  import sys
  from os import path
  sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
