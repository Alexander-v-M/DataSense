from DataSense.elements.models.lstm import LSTMModel
from DataSense.elements.models.gru import GRUModel
from DataSense.elements.models.rnn import RNNModel

from typing import Dict
from torch.nn import Module


def get_model(model: str, model_params: Dict) -> Module:
    """
    Create and return an instance of the specified neural network model.

    Args:
        model (str): The name of the neural network model ('rnn', 'lstm', or 'gru').
        model_params (dict): Dictionary of model-specific parameters.

    Returns:
        nn.Module: An instance of the specified neural network model.
    """
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)
