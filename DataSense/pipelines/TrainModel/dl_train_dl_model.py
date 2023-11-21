from pandas import DataFrame
from pprint import pprint
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from DataSense.elements.feature_creation import get_features
from DataSense.elements.get_model import get_model
from DataSense.elements.metric import calculate_metrics, calc_residuals
from DataSense.elements.postprocess import format_predictions, remove_outliers
from DataSense.elements.util import get_scaler
from DataSense.elements.visualize import plot_dataset, plot_predictions, plot_residuals
from DataSense.elements.dataloaders import f_to_tvt_loaders
from DataSense.elements.util import train_val_test_split
from DataSense.elements.save_meta_data import save_meta_dl

from DataSense.pipelines.TrainModel.cls_opt_uni_v import Optimization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(data: DataFrame,
                features: str,
                batch_size: int,
                hidden_dim: int,
                layer_dim: int,
                dropout: float,
                n_epochs: int,
                learning_rate: float,
                model_type: str,
                weight_decay: float = 1e-6,
                output_dim: int = 1,
                n_lag: int = None,
                vis_data: bool = True,
                scaler_in: str = 'minmax',
                give_metr: bool = False):
    """
    Train a neural network model on the provided data.

    Args:
        data (DataFrame): The input data as a DataFrame.
        features (str): The version of features to use.
        batch_size (int): Batch size for training and validation.
        hidden_dim (int): Dimension of hidden layers in the neural network.
        layer_dim (int): Number of layers in the neural network.
        dropout (float): Dropout probability for regularization.
        n_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimization.
        model_type (str): type of nural network to use; gru, rnn, lstm
        weight_decay (float, optional): L2 regularization weight (default is 1e-6).
        output_dim (int, optional): Dimension of the output layer (default is 1).
        n_lag (int, optional): Number of lag features to consider, use if features = 'generated' (default is None).
        vis_data (bool, optional): Whether to visualize the original data (default is True).
        scaler_in (str): scaler methode
    """

    # Get DataFrame
    df = data

    if vis_data:
        # Visualize the dataset
        plot_dataset(df, title='original data, weekly aggregated')

    # Extract relevant features from the DataFrame
    df_features = get_features(df=df, version=features, n_lag=n_lag)

    # Normalize the features using a given scaler
    scaler = get_scaler(scaler_in)

    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, 'value', 0.2)

    # Create data loaders for training and evaluation
    train_loader, val_loader, test_loader, test_loader_one, scaler_fitted = f_to_tvt_loaders(
        datasets=(X_train, X_val, X_test, y_train, y_val, y_test),
        scaler=scaler,
        batch_size=batch_size)

    # Determine the input dimension for the model
    input_dim = len(X_train.columns)

    # Define model parameters
    model_params = {'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'layer_dim': layer_dim,
                    'output_dim': output_dim,
                    'dropout_prob': dropout}

    # Create the model
    model = get_model(model_type, model_params)

    # Define the loss function (Mean Squared Error)
    loss_fn = nn.MSELoss(reduction="mean")

    # Define the optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize the optimization process
    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)

    # Train the model
    state_dict = opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)

    # Plot training and validation losses
    opt.plot_losses()

    # Evaluate the model on the test data
    predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)

    # Format the predictions and values
    df_result = format_predictions(predictions, values, X_test, scaler)

    # Calculate evaluation metrics
    result_metrics = calculate_metrics(df_result)
    pprint(result_metrics)

    # Residuals
    residuals, mean_res = calc_residuals(df_result)
    plot_residuals(residuals)

    # Plot the model's predictions
    plot_predictions(df_result)

    if give_metr:
        return calculate_metrics(df_result, lis=True)
    else:
        # return the trained model file path
        return save_meta_dl(state_dict=state_dict,
                            model_type=model_type,
                            hidden_dim=hidden_dim,
                            layer_dim=layer_dim,
                            dropout=dropout,
                            output_dim=output_dim,
                            input_dim=input_dim,
                            batch_size=batch_size,
                            n_epochs=n_epochs,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            n_lag=n_lag,
                            opt=opt,
                            result_metrics=result_metrics,
                            scaler_fitted=scaler_fitted,
                            scaler_in=scaler_in), df_result
