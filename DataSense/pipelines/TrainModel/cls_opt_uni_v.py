import numpy as np
import torch
from tqdm import tqdm

from DataSense.elements.visualize import plot_losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        """
        Initialize the Optimization class.

        Args:
            model: The neural network model to be optimized.
            loss_fn: The loss function used for optimization.
            optimizer: The optimization algorithm (e.g., Adam).

        Attributes:
            model: The neural network model.
            loss_fn: The loss function.
            optimizer: The optimization algorithm.

        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

        self.best_model = None
        self.best_epoch = None
        self.best_val_loss = np.inf

    def train_step(self, x, y):
        """
        Perform a single training step.

        Args:
            x: The input data.
            y: The target output data.

        Returns:
            float: The loss for this training step.
        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def evaluate(self, test_loader,
                 batch_size: int = 1,
                 n_features: int = 1):
        """
        Evaluate the model on a test dataset.

        Args:
            test_loader: DataLoader for the test dataset.
            batch_size: Batch size for evaluation (default is 1).
            n_features: Number of features in the input data (default is 1).

        Returns:
            tuple: A tuple containing predictions and true values.
        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().cpu().numpy())
                values.append(y_test.to(device).detach().cpu().numpy())

        return predictions, values

    def train(self, train_loader,
              val_loader,
              batch_size: int = 64,
              n_epochs: int = 50,
              n_features: int = 1):
        """
        Train the model.

        Args:
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            batch_size: Batch size for training (default is 64).
            n_epochs: Number of training epochs (default is 50).
            n_features: Number of features in the input data (default is 1).

        Returns:
            None
        """

        # loading bar
        pbar = tqdm(total=n_epochs, desc="Training")

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

                if validation_loss < self.best_val_loss:
                    self.best_val_loss = validation_loss
                    self.best_model = self.model
                    self.best_epoch = epoch

            pbar.set_postfix_str(f"TrainLoss: {training_loss:.4f}; ValLoss: {validation_loss:.4f};"
                                 f" BestValLoss: {self.best_val_loss}; bestEpoch: {self.best_epoch}")
            pbar.update()

        return self.best_model.state_dict

    def plot_losses(self):
        """
        Plot the training and validation losses.

        Returns:
            None
        """

        plot_losses(self.train_losses, self.val_losses, self.best_epoch, self.best_val_loss)

