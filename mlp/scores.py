# -*- coding: utf-8 -*-
import torch

"""Validation accuracy
"""
def evaluate(net, data_loader, criterion, device, topk=(1,)):
  """Inference with net on the given test set
  """
  maxk = max(topk)
  correct_top1_cnt, correct_top5_cnt, avg_loss = 0, 0, 0
  total_count = 0
  
  net = net.to(device=device)
  net = net.eval()
  
  with torch.no_grad():
    for _, (x, target) in enumerate(data_loader):
      x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
      out = net(x)
      loss = criterion(out, target)
      
      _, pred = out.data.topk(5, 1, True, True)
      pred = pred.t()
      correct = pred.eq(target.view(1, -1).expand_as(pred))
      
      total_count += x.data.size()[0]
      correct_top1_cnt += correct[:1].view(-1).float().sum(0, keepdim=True)
      correct_top5_cnt += correct[:5].view(-1).float().sum(0, keepdim=True)
      # smooth average
      avg_loss += loss.item()
    avg_loss /= len(data_loader)
    top1_acc = correct_top1_cnt.mul_(100.0 / total_count)
    top5_acc = correct_top5_cnt.mul_(100.0 / total_count)

  return avg_loss, top1_acc.item(), top5_acc.item()
  
def compute_regularization_loss(net, weight_decay):
  """ Compute the weight decay term for net, with scaling coefficient weight_decay
  
      The L2 regularization is computed according to the weight decay formulation
      in torch.optim.SGD:
      
      https://github.com/pytorch/pytorch/blob/a3410f7994338279b12054756585cfe52689de4d/torch/optim/sgd.py#L93
  """
  l2_reg = 0
  for name, W in net.named_parameters():
    if 'weight' in name:
      l2_reg += W.norm(2)
  l2_reg *= weight_decay
  return l2_reg
  
