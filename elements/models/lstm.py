import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """
        Initialize an LSTM-based model.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden state.
            layer_dim (int): Number of LSTM layers.
            output_dim (int): Dimension of the output.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        ).to(device)

        # Define the fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (tensor): Input data tensor.

        Returns:
            tensor: Output tensor from the model.
        """
        # Initialize hidden state for the first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state for the first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Detach hidden and cell states to enable truncated backpropagation through time
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshape the outputs to the shape of (batch_size, seq_length, hidden_size)
        # to prepare for the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to the desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
