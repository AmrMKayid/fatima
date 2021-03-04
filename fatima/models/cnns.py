import torch
import torch.nn as nn

from fatima.models.base import Model


class CNN(Model):

  def __init__(
      self,
      config,
      name: str = "CNNModel",
      *,
      input_dim: tuple = (32, 32),
      tau: int = 3,
      output_shape: int = 1,
  ) -> None:
    super(CNN, self).__init__(config, name)

    self.input_dim = input_dim
    self.tau = tau
    self.output_shape = output_shape

    self.cnn = nn.Sequential(
        nn.Conv2d(tau, 32, kernel_size=8),
        nn.ReLU(),
        nn.Conv2d(32, 16, kernel_size=4),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3),
        nn.ReLU(),
    )

    self.fc_inputs = self.feature_size(input_dim)

    self.fc = nn.Sequential(
        nn.Linear(self.fc_inputs, 512, bias=True),
        nn.ReLU(),
        nn.Linear(512, output_shape),
    )

    self.sm = nn.Sigmoid()

  def feature_size(self, input_dim):
    return self.cnn(torch.zeros(1, self.tau, *input_dim)).flatten().shape[0]

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    states_features = self.cnn(inputs).reshape(-1, self.fc_inputs)
    return self.sm(self.fc(states_features))
