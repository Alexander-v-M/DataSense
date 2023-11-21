import pickle
import pandas as pd

from DataSense.elements.visualize import plot_armima_forecasting


class Predictor:
    def __init__(self, model_storage_fp, dataset, num_prediction, model_type):
        """
        Initialize a Predictor object for making time series forecasts using a pre-trained model.

        Args:
            model_storage_fp (str): Filepath to the directory containing the pre-trained model.
            dataset (DataFrame): The dataset to use for forecasting.
            num_prediction (int): The number of future time points to forecast.
            model_type (str): The type or name of the forecasting model.

        Note:
            The pre-trained model should be saved as a pickle file in the 'model_storage_fp' directory.
        """
        self.ms_fp = model_storage_fp
        self.dataset = dataset
        self.num_prediction = num_prediction
        self.model_type = model_type

        with open(rf'{self.ms_fp}\model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def save_forecast_to_csv(self, df):
        """
        Save the forecasted data to a CSV file.

        Args:
            df (DataFrame): The forecasted data to be saved.

        Note:
            The forecasted data is saved as a CSV file in the model storage directory.
        """
        df.to_csv(fr'{self.ms_fp}\forecast.csv')

    def forecast(self):
        """
        Generate forecasts for future time points and save them to a CSV file.

        """
        # Forecast future values
        fitted, confint = self.model.predict(n_periods=self.num_prediction, return_conf_int=True)
        index_of_fc = pd.date_range(self.dataset.index[-1], periods=self.num_prediction, freq='W')

        # Create series for plotting purposes
        fitted_series = pd.Series(fitted, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        self.save_forecast_to_csv(fitted_series)

        plot_armima_forecasting(fitted_series, lower_series, upper_series, self.dataset, model_type=self.model_type)

