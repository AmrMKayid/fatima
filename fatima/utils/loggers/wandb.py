"""Utilities for logging to Weights & Biases."""

import wandb
from absl import flags
from ray.tune.integration.wandb import WandbLogger

from fatima.utils.loggers import base

wandb.init(project="goodai-il-breakout", config=flags.FLAGS)


class WandBLogger(base.Logger, WandbLogger):
  """Logs to a `wandb` dashboard."""

  def write(self, values: base.LoggingData) -> None:
    wandb.log(values)
