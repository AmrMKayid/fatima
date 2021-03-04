import os
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class UpsideDownDataset(torchvision.datasets.CIFAR10):

  def __init__(
      self,
      root='./data',
      download=True,
      train=True,
      transform: Callable = transforms.Compose([
          transforms.ToTensor(),
      ]),
  ) -> None:
    super().__init__(root=root, download=download, train=train)

    self._transform = transform
    self.flip_transform = transforms.RandomVerticalFlip(p=1.0)
    self.classes = ['up', 'down']
    self.class_to_idx = {'up': 0, 'down': 1}

  def __getitem__(self, index):
    im, label = super().__getitem__(index)
    if index % 2 == 0:
      im, label = self.flip_transform(im), 1
    else:
      label = 0
    return self._transform(im), label


def get_dls(config):

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
      )
  ])

  train_dataset = UpsideDownDataset(
      root=config.data_path,
      train=True,
      download=True,
      transform=transform,
  )

  train_val_sizes = [
      int(len(train_dataset) * .8),
      int(len(train_dataset) - int(len(train_dataset) * .8))
  ]

  train_dataset, val_dataset = torch.utils.data.random_split(
      train_dataset,
      train_val_sizes,
  )
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=config.batch_size,
      shuffle=True,
  )
  val_dataloader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=config.batch_size * 5,
      shuffle=True,
  )

  test_dataset = UpsideDownDataset(
      root=config.data_path,
      train=False,
      download=True,
      transform=transform,
  )
  test_dataloader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=config.batch_size,
      shuffle=False,
      num_workers=config.num_workers,
  )

  return SimpleNamespace(
      train=train_dataloader,
      val=val_dataloader,
      test=test_dataloader,
  )
