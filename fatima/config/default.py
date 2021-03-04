import json
import os
import time

import torch
from loguru import logger

from fatima.config.base import Config


def defaults() -> Config:
  config = Config(
      device='cuda' if torch.cuda.is_available() else 'cpu',
      data_path="./data",
      output_dir='output',
      epochs=5,
      lr=1e-2,
      momentum=0.9,
      betas=(0.9, 0.999),
      batch_size=512,
      weight_decay=0.00001,
      max_timesteps=1000,
      dropout=0.5,
      seed=1337,
      num_workers=4,
      save_model_frequency=4,
      eval_episodes=500,
  )

  # Creates the necessary output directory.
  config.output_dir = os.path.abspath(config.output_dir)
  os.makedirs(config.output_dir, exist_ok=True)
  logger.debug(f'Creating output dir `{config.output_dir}`')

  config.run_exp_folder = os.path.join(
      config.output_dir, "Exp_" + time.strftime('%Y-%m-%d_%H%M%S'))
  os.makedirs(config.run_exp_folder, exist_ok=True)
  logger.debug(f'Creating run exp folder `{config.run_exp_folder}`')

  config.log_dir = os.path.join(config.run_exp_folder, "logs")
  os.makedirs(config.log_dir, exist_ok=True)

  config.ckpt_dir = os.path.join(config.run_exp_folder, "ckpts")
  os.makedirs(config.ckpt_dir, exist_ok=True)
  config.model_path = os.path.abspath(os.path.join(config.ckpt_dir, 'model.th'))

  # Save the configuration in a config.json file
  with open(os.path.join(config.run_exp_folder, 'config.json'), 'w') as f:
    json.dump(vars(config), f, indent=2, default=lambda o: o.__dict__)
  logger.info('Saving configuration file in `{0}`'.format(
      os.path.abspath(os.path.join(config.run_exp_folder, 'config.json'))))

  config.device = torch.device(config.device)

  return config
