import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """
        Initialize a GRU-based model.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden state.
            layer_dim (int): Number of GRU layers.
            output_dim (int): Dimension of the output.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(GRUModel, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # Define the GRU layer
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        ).to(device)

        # Define the fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        """
        Forward pass of the GRU model.

        Args:
            x (tensor): Input data tensor.

        Returns:
            tensor: Output tensor from the model.
        """

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Forward pass through the GRU layer
        out, _ = self.gru(x, h0.detach())

        # Extract the output at the last time step
        out = out[:, -1, :]

        # Pass through the fully connected output layer
        out = self.fc(out)

        return out
