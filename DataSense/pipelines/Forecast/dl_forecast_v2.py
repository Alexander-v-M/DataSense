import json
from pickle import load

import numpy as np
import pandas as pd
import torch

from DataSense.elements.feature_creation import get_features
from DataSense.elements.get_model import get_model
from DataSense.elements.load_data import get_data_wf
from DataSense.elements.postprocess import inverse_transform
from DataSense.elements.util import train_val_test_split


# class Predictor:
#
#     def __init__(self, model_storage_fp, dataset, num_prediction):
#         self.ms_fp = model_storage_fp
#         self.dataset = dataset
#         self.num_prediction = num_prediction
#
#         self.model = None
#         self.scaler = None
#         self.feature_len = None
#
#         with open(rf'{self.ms_fp}\meta_data.json') as f:
#             self.meta_data = json.load(f)
#
#         self.remake_model()
#
#     def remake_model(self):
#         model_meta_data = self.meta_data['model']
#
#         model_params = {'input_dim': int(model_meta_data['input_dim']),
#                         'hidden_dim': int(model_meta_data['hidden_dim']),
#                         'layer_dim': int(model_meta_data['layer_dim']),
#                         'output_dim': int(model_meta_data['output_dim']),
#                         'dropout_prob': float(model_meta_data['dropout'])}
#
#         # Create the model
#         self.model = get_model(model_meta_data['model_type'], model_params)
#         self.model.load_state_dict(torch.load(rf'{self.ms_fp}\model'))
#
#     def get_scaled_data(self):
#         # Extract relevant features from the DataFrame
#         df_features = get_features(df=self.dataset,
#                                    version='generated',
#                                    n_lag=self.meta_data['n_lag'])
#
#         self.feature_len = len(df_features.columns) - 1
#
#         self.scaler = load(open(rf'{self.ms_fp}\scaler.pkl', 'rb'))
#
#         _, _, _, _, _, y_test = train_val_test_split(df_features, 'value', 0.2)
#
#         return self.scaler.transform(y_test)
#
#     def predict(self, look_back, data_scaled):
#
#         if look_back is None:
#             look_back = self.feature_len
#
#         data_scaled = np.array(data_scaled).reshape((-1))
#         prediction_list = data_scaled[-look_back:]
#
#         with torch.no_grad():
#             self.model.eval()
#             for _ in range(self.num_prediction):
#                 x = prediction_list[-look_back:]
#                 x = torch.tensor(x, dtype=torch.float32).view(1, 1, self.feature_len).to('cuda')
#                 yhat = self.model(x)
#                 prediction_list = np.append(prediction_list, yhat.to('cuda').detach().cpu().numpy())
#             prediction_list = prediction_list[look_back - 1:]
#
#         return prediction_list
#
#     def save_forecast_to_csv(self, df):
#         df.to_csv(fr'{self.ms_fp}\forecast.csv')
#
#     def predict_dates(self):
#         last_date = self.dataset.index[-1]
#         prediction_dates = pd.date_range(last_date, freq='W',
#                                          periods=self.num_prediction + 1).tolist()  # periods=self.num_prediction + 1
#         return prediction_dates
#
#     def forecast(self):
#         # Make predictions and get prediction dates
#         forecast = self.predict(self.feature_len, self.get_scaled_data())
#         forecast_dates = self.predict_dates()
#
#         df_p = pd.DataFrame(forecast)
#         df_p.index = forecast_dates
#         df_p.sort_index()
#         df_p.columns = ['value']
#         df_p = inverse_transform(self.scaler, df_p, [["value"]])
#
#         self.save_forecast_to_csv(df_p)
#
#         return df_p


class Predictor:
    def __init__(self, model_storage_fp, dataset, num_prediction):
        """
        Initialize a Predictor object for making predictions using a trained model.

        Args:
            model_storage_fp (str): Filepath to the directory containing the trained model and related files.
            dataset (DataFrame): The dataset to use for prediction.
            num_prediction (int): The number of future predictions to generate.

        Note:
            The model and scaler should be saved in the 'model_storage_fp' directory.
        """
        self.ms_fp = model_storage_fp
        self.dataset = dataset
        self.num_prediction = num_prediction
        self.model = None
        self.scaler = None
        self.feature_len = None

        with open(rf'{self.ms_fp}\meta_data.json') as f:
            self.meta_data = json.load(f)

        self.remake_model()

    def remake_model(self):
        """
        Load the trained model from the model storage directory and create an instance of it.

        Note:
            This method loads the model's architecture and state dictionary and creates an instance of the model.
        """
        model_meta_data = self.meta_data['model']

        model_params = {'input_dim': int(model_meta_data['input_dim']),
                        'hidden_dim': int(model_meta_data['hidden_dim']),
                        'layer_dim': int(model_meta_data['layer_dim']),
                        'output_dim': int(model_meta_data['output_dim']),
                        'dropout_prob': float(model_meta_data['dropout'])}

        # Create the model
        self.model = get_model(model_meta_data['model_type'], model_params)
        self.model.load_state_dict(torch.load(rf'{self.ms_fp}\model'))

    def get_scaled_data(self):
        """
        Load and scale the data for prediction.

        Returns:
            np.ndarray: Scaled data ready for prediction.
        """
        # Extract relevant features from the DataFrame
        df_features = get_features(df=self.dataset,
                                   version='generated',
                                   n_lag=self.meta_data['n_lag'])

        self.feature_len = len(df_features.columns) - 1

        self.scaler = load(open(rf'{self.ms_fp}\scaler.pkl', 'rb'))

        _, _, _, _, _, y_test = train_val_test_split(df_features, 'value', 0.2)

        return self.scaler.transform(y_test)

    def predict(self, look_back, data_scaled):
        """
        Generate predictions using the trained model.

        Args:
            look_back (int): The number of previous values to use as input for generating predictions.
            data_scaled (np.ndarray): Scaled data for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        if look_back is None:
            look_back = self.feature_len

        data_scaled = np.array(data_scaled).reshape((-1))
        prediction_list = data_scaled[-look_back:]

        with torch.no_grad():
            self.model.eval()
            for _ in range(self.num_prediction):
                x = prediction_list[-look_back:]
                x = torch.tensor(x, dtype=torch.float32).view(1, 1, self.feature_len).to('cuda')
                yhat = self.model(x)
                prediction_list = np.append(prediction_list, yhat.to('cuda').detach().cpu().numpy())
            prediction_list = prediction_list[look_back - 1:]

        return prediction_list

    def save_forecast_to_csv(self, df):
        """
        Save the forecasted data to a CSV file.

        Args:
            df (DataFrame): The forecasted data.

        Note:
            The forecasted data is saved as a CSV file in the model storage directory.
        """
        df.to_csv(fr'{self.ms_fp}\forecast.csv')

    def predict_dates(self):
        """
        Generate prediction dates based on the dataset's last date and the number of predictions.

        Returns:
            List of datetime: Prediction dates.
        """
        last_date = self.dataset.index[-1]
        prediction_dates = pd.date_range(last_date, freq='W',
                                         periods=self.num_prediction + 1).tolist()  # periods=self.num_prediction + 1
        return prediction_dates

    def forecast(self):
        """
        Make predictions and save the forecasted data to a CSV file.

        Returns:
            DataFrame: Forecasted data.
        """
        # Make predictions and get prediction dates
        forecast = self.predict(self.feature_len, self.get_scaled_data())
        forecast_dates = self.predict_dates()

        df_p = pd.DataFrame(forecast)
        df_p.index = forecast_dates
        df_p.sort_index()
        df_p.columns = ['value']
        df_p = inverse_transform(self.scaler, df_p, [["value"]])

        self.save_forecast_to_csv(df_p)

        return df_p
