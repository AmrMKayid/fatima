import torch


class Agent:

  def __init__(
      self,
      *,
      config,
      name: str,
  ) -> None:
    self._config = config
    self._name = name

  def act(
      self,
      state,
  ) -> torch.Tensor:
    """Select an action to take.

    Args:
      state: a Tensor of states.
    Returns:
      action: a Tensor of the selected action.
    """
    raise NotImplementedError

  @property
  def config(self):
    return self._config


class BaseTrainer:

  def __init__(
      self,
      *,
      config,
      name: str,
  ) -> None:
    self._config = config
    self._name = name

  def train(
      self,
      dataloader,
  ) -> torch.Tensor:
    """Select an action to take.

    Args:
      state: a Tensor of states.
    Returns:
      action: a Tensor of the selected action.
    """
    raise NotImplementedError

  def test(
      self,
      dataloader,
  ) -> torch.Tensor:
    """Select an action to take.

    Args:
      state: a Tensor of states.
    Returns:
      action: a Tensor of the selected action.
    """
    raise NotImplementedError

  @property
  def config(self):
    return self._config
