import torch
import torch.nn as nn
import torchvision

from fatima.models.base import Model


class ResNet50(Model):

  def __init__(
      self,
      config,
      name: str = "ResNet50Model",
      *,
      input_dim: tuple = (32, 32),
      tau: int = 3,
      output_shape: int = 1,
  ) -> None:
    super(ResNet50, self).__init__(config, name)

    self.input_dim = input_dim
    self.tau = tau
    self.output_shape = output_shape

    self.resnet = torchvision.models.resnet50(pretrained=True)

    self.resnet.fc = nn.Sequential(
        nn.Linear(2048, self.output_shape, bias=True),
        nn.Sigmoid(),
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    return self.resnet(inputs)


class ResNet18(Model):

  def __init__(
      self,
      config,
      name: str = "ResNet18Model",
      *,
      input_dim: tuple = (32, 32),
      tau: int = 3,
      output_shape: int = 1,
  ) -> None:
    super(ResNet18, self).__init__(config, name)

    self.input_dim = input_dim
    self.tau = tau
    self.output_shape = output_shape

    self.resnet = torchvision.models.resnet18(pretrained=True)

    self.resnet.fc = nn.Sequential(
        nn.Linear(512, self.output_shape, bias=True),
        nn.Sigmoid(),
    )

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    return self.resnet(inputs)
