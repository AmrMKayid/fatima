import argparse


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      "Imitation Learning from recorded behavior of an expert! 🕹👾")

  parser.add_argument(
      '--data_path',
      default="./data",
      type=str,
      help='The path to IL data',
  )

  parser.add_argument(
      '--num_epochs',
      default=25,
      type=int,
      help='The number of epochs to run training for.',
  )
  parser.add_argument(
      '--lr',
      default=1e-2,
      type=float,
      help='Hyperparameter: learning rate.',
  )
  parser.add_argument(
      '--output_dir',
      default='output',
      type=str,
      help='Hyperparameter: feature reg',
  )
  parser.add_argument(
      '--batch_size',
      default=512,
      type=int,
      help='Hyperparameter: batch size.',
  )
  parser.add_argument(
      '--weight_decay',
      default=0.00001,
      type=float,
      help='Hyperparameter: Optimizer decay rate',
  )
  parser.add_argument(
      '--dropout',
      default=0.5,
      type=float,
      help='Hyperparameter: Dropout rate',
  )
  parser.add_argument(
      '--seed',
      default=1337,
      type=int,
      help='seed number',
  )
  parser.add_argument(
      '--num_workers',
      default=4,
      type=int,
      help='Number of workers for dataloading...',
  )
  parser.add_argument(
      '--save_model_frequency',
      default=4,
      type=int,
      help='Frequency to save the trained model',
  )
  parser.add_argument(
      '--debug',
      default=False,
      type=bool,
      help='Debug mode. Log additional things',
  )

  FLAGS = parser.parse_args()
  return FLAGS
