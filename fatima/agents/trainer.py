import torch
from tqdm import tqdm

from fatima.models.saver import Checkpointer
from fatima.utils.loggers.tensorboard import TensorBoardLogger

from .base import BaseTrainer


class Trainer(BaseTrainer):

  def __init__(self, *, config, name, model, dls):
    super().__init__(config=config, name=name)

    self.dls = dls
    self.model = model.to(config.device)
    self.optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    self.criterion = torch.nn.BCELoss()

    self.writer = TensorBoardLogger(log_dir=config.log_dir)
    self.checkpointer = Checkpointer(model=self.model, ckpt_dir=config.ckpt_dir)

  def run(self):
    num_epochs = self.config.epochs

    with tqdm(range(num_epochs)) as pbar_epoch:
      for epoch in pbar_epoch:
        # Trains model on whole training dataset, and writes on `TensorBoard`.
        loss_train, accuracy_train = self.train(self.dls.train)
        self.writer.write({
            "loss_train": loss_train.detach().cpu().numpy().item(),
            "accuracy_train": accuracy_train.detach().cpu().numpy().item(),
            "global_step": epoch,
        })

        # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
        loss_val, accuracy_val = self.train(self.dls.val, mode='eval')
        self.writer.write({
            "loss_val": loss_val.detach().cpu().numpy().item(),
            "accuracy_val": accuracy_val.detach().cpu().numpy().item(),
            "global_step": epoch,
        })

        # Checkpoints model weights.
        if epoch % self.config.save_model_frequency == 0:
          self.checkpointer.save(epoch)

        # Updates progress bar description.
        pbar_epoch.set_description(
            "[{}] Train Loss: {:.2f} | Eval Loss: {:.2f} | Train Accuracy: {:.2f} | Eval Accuracy: {:.2f}"
            .format(epoch,
                    loss_train.detach().cpu().numpy().item(),
                    loss_val.detach().cpu().numpy().item(),
                    accuracy_train.detach().cpu().numpy().item(),
                    accuracy_val.detach().cpu().numpy().item()))

  def train(self, dataloader, mode='train'):
    self.model.train()
    total_loss, losses = 0.0, []
    total_accuracy, accuracies = 0.0, []

    with tqdm(dataloader) as pbar:
      for batch in pbar:
        images, labels = map(
            lambda data: data.to(self.config.device).float(),
            batch,
        )

        # Resets optimizer's gradients.
        self.optimizer.zero_grad()

        # Forward pass from the model.
        preds = self.model(images).squeeze()

        # Calculates loss (CrossEntropy).
        if mode == 'train':
          loss = self.criterion(preds, labels)

          # Backward pass.
          loss.backward()

          # Performs a gradient descent step.
          self.optimizer.step()
        else:
          with torch.no_grad():
            loss = self.criterion(preds, labels)

        losses.append(loss.data)

        accuracy = self.accuracy(preds, labels)
        accuracies.append(accuracy)

        total_loss += loss
        total_accuracy += accuracy

        pbar.set_description(
            f"[{mode}] loss = {sum(losses) * (1 / len(losses))} | accuracy = {sum(accuracies) * (1 / len(accuracies))}"
        )

    return (
        total_loss / len(dataloader),
        total_accuracy / len(dataloader),
    )

  def test(self, dataloader):
    total_accuracy, accuracies = 0.0, []

    with tqdm(dataloader) as pbar:
      with torch.no_grad():
        for batch in pbar:
          images, labels = map(
              lambda data: data.to(self.config.device).float(),
              batch,
          )

          # Forward pass from the model.
          preds = self.model(images).squeeze()
          accuracy = self.accuracy(preds, labels)
          accuracies.append(accuracy)

          total_accuracy += accuracy

          pbar.set_description(
              f"Test Accuracy = {sum(accuracies) * (1 / len(accuracies))}")

    return total_accuracy / len(dataloader)

  def accuracy(self, outputs, targets):
    """Computes the accuracy for multiple binary predictions."""
    pred = outputs >= 0.5
    truth = targets >= 0.5
    acc = pred.eq(truth).sum() / targets.numel()
    return (acc * 100)
