import json
import os
from datetime import datetime
from pathlib import Path
import torch
from pickle import dump


def make_and_get_model_dir(model):
    """
    Create and return a directory path for storing a trained model based on the current timestamp.

    Args:
        model (str): The model name or identifier that will be used to create a subdirectory within the 'trained_models'
                     directory.

    Returns:
        str: The absolute path to the newly created model directory.
    """
    working_dir = os.path.join(os.path.abspath(__file__ + "/../../"), 'trained_models')
    storage_dir = os.path.join(working_dir, datetime.now().strftime(f"{model}%Y-%m-%d_%H_%M_%S"))
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    return str(storage_dir)


def save_meta_dl(state_dict, result_metrics, **kwargs):
    """
    Save model metadata, hyperparameters, metrics, and state dictionary to a directory.

    This function saves various pieces of information related to a deep learning model, including its metadata,
    hyperparameters, evaluation metrics, and the model's state dictionary, into a designated directory.

    Args:
        state_dict: The state dictionary of the trained deep learning model.
        result_metrics: Metrics or evaluation results of the model.
        **kwargs: Additional keyword arguments containing model configuration and hyperparameters.

    Returns:
        str: The absolute path to the directory where the metadata and model artifacts are saved.
    """
    fp = make_and_get_model_dir(model=kwargs['model_type'])
    os.chdir(fp)

    data = {
        'model': {
            'model_type': kwargs['model_type'],
            'hidden_dim': kwargs['hidden_dim'],
            'layer_dim': kwargs['layer_dim'],
            'dropout': kwargs['dropout'],
            'output_dim': kwargs['output_dim'],
            'input_dim': kwargs['input_dim']
        },
        'batch_size': kwargs['batch_size'],
        'n_epochs': kwargs['n_epochs'],
        'learning_rate': kwargs['learning_rate'],
        'weight_decay': kwargs['weight_decay'],
        'n_lag': kwargs['n_lag'],
        'scaler_in': kwargs['scaler_in']
    }

    with open('meta_data.json', 'w') as write_file:
        json.dump(data, write_file)

    with open('metrics.txt', 'w') as f:
        f.write(result_metrics)

    torch.save(state_dict(), 'model')

    dump(kwargs['scaler_fitted'], open('scaler.pkl', 'wb'))

    return fp


def save_meta_ml(model, model_summary, result_metrics, **kwargs):
    """
    Save metadata, hyperparameters, model summary, and metrics to a directory for machine learning models.

    This function saves various pieces of information related to a machine learning model, including its metadata,
    hyperparameters, model summary, and evaluation metrics, into a designated directory.

    Args:
        model: The trained machine learning model to be saved.
        model_summary: A summary or description of the machine learning model.
        result_metrics (str): Metrics or evaluation results of the model.
        **kwargs: Additional keyword arguments containing model configuration and hyperparameters.

    Returns:
        str: The absolute path to the directory where the metadata and model artifacts are saved.
    """
    fp = make_and_get_model_dir(model=kwargs['model_type'])
    os.chdir(fp)

    data = {
        'model': {
            'start_p': kwargs['start_p'],
            'start_q': kwargs['start_q'],
            'max_p': kwargs['max_p'],
            'max_q': kwargs['max_q'],
            'm': kwargs['m'],
            'start_P': kwargs['start_P'],
            'D': kwargs['D'],
        },
        'seasonal': kwargs['seasonal'],
    }

    with open('meta_data.json', 'w') as write_file:
        json.dump(data, write_file)

    with open('model_summary.txt', 'w') as f:
        f.write(str(model_summary))

    with open('metrics.txt', 'w') as f:
        f.write(result_metrics)

    dump(model, open('model.pkl', 'wb'))

    return fp


def save_txt(string, model):
    """
    Save a string for the experimental design to a text file within a designated directory.

    Args:
        string (str): The text content to be saved to the file.
        model (str): The model or purpose associated with the directory where the text file will be saved.

    Returns:
        None
    """
    fp = make_and_get_model_dir(model=model)
    os.chdir(fp)

    with open('doe.txt', 'w') as f:
        f.write(string.to_string())
