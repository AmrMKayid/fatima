import random

import numpy as np
import torch
from loguru import logger


def init_random_seeds(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    logger.debug("PyTorch is using Cuda...")
    torch.cuda.manual_seed_all(seed)
  else:
    logger.debug("PyTorch is using CPU...")
