import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple


def f_to_tvt_loaders(scaler, batch_size: int, datasets: Tuple) -> Tuple:
    """
    Create data loaders for training, validation, and testing using the provided scaler.

    Args:
        scaler: Scaler object for normalizing data.
        batch_size (int): Batch size for data loaders.
        datasets (tuple): A tuple containing training, validation, and test datasets.

    Returns:
        tuple: A tuple of data loaders for training, validation, and testing,
        and a testing data loader with batch size 1.
    """
    # Unpack datasets
    x_train, x_val, x_test, y_train, y_val, y_test = datasets

    # Scale the features and targets using the scaler
    x_train_arr = scaler.fit_transform(x_train)
    x_val_arr = scaler.transform(x_val)
    x_test_arr = scaler.transform(x_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)

    # Convert scaled data to PyTorch tensors
    train_features = torch.Tensor(x_train_arr)
    train_targets = torch.Tensor(y_train_arr)
    val_features = torch.Tensor(x_val_arr)
    val_targets = torch.Tensor(y_val_arr)
    test_features = torch.Tensor(x_test_arr)
    test_targets = torch.Tensor(y_test_arr)

    # Create TensorDatasets from PyTorch tensors
    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    # Create data loaders for training, validation, and testing
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, test_loader_one, scaler
