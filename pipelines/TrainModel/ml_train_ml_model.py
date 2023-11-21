import pandas as pd
import pmdarima as pm
from matplotlib import pyplot as plt
from datetime import datetime

from DataSense.elements.metric import calculate_metrics_no_df
from DataSense.elements.visualize import plot_armima_prediction
from DataSense.elements.save_meta_data import save_meta_ml
from DataSense.elements.util import train_val_test_split


def train_model(data: pd.DataFrame,
                seasonal: bool,
                start_p: int,
                start_q: int,
                max_p: int,
                max_q: int,
                m: int,
                start_P: int):
    """
    Train a time series forecasting model using ARIMA or SARIMA based on the specified parameters.

    Parameters:
        data (pd.DataFrame): Time series data to be used for training the model.
        seasonal (bool): Flag indicating whether to use seasonal decomposition (SARIMA) or not (ARIMA).
        start_p (int): Initial value for the 'p' parameter in ARIMA order selection.
        start_q (int): Initial value for the 'q' parameter in ARIMA order selection.
        max_p (int): Maximum value for the 'p' parameter in ARIMA order selection.
        max_q (int): Maximum value for the 'q' parameter in ARIMA order selection.
        m (int): Seasonal periodicity (number of periods in a season).
        start_P (int): Initial value for the 'P' parameter in seasonal order selection.

    Returns:
        dict: A dictionary containing model information and metrics.
    """

    _, _, y_train, y_test = train_val_test_split(data, validation=False, test_ratio=0.2, target_col='value')

    # Determine the model type (ARIMA or SARIMA) and seasonal parameters
    if seasonal:
        s, S, model_type = 1, True, 'SARIMA'
    else:
        s, S, model_type = 0, False, 'ARIMA'

    # Train the ARIMA or SARIMA model using auto_arima from pmdarima
    smodel = pm.auto_arima(y_train, start_p=start_p, start_q=start_q,
                           test='adf',
                           max_p=max_p, max_q=max_q, m=m,
                           start_P=start_P, seasonal=S,
                           d=None, D=s, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

    # Forecast future values
    fitted, confint = smodel.predict(n_periods=int(len(y_test)), return_conf_int=True)

    # Create series for plotting purposes
    fitted_series = pd.Series(fitted.values, index=y_test.index)
    lower_series = pd.Series(confint[:, 0], index=y_test.index)
    upper_series = pd.Series(confint[:, 1], index=y_test.index)

    result_metrics = calculate_metrics_no_df(fitted_series, y_test)

    # Plot the ARIMA/SARIMA predictions and forecasting
    plot_armima_prediction(fitted_series, lower_series, upper_series, data, model_type=model_type)

    # Save model information and metrics in a dictionary
    return save_meta_ml(
        model=smodel,
        model_summary=smodel.summary(),
        result_metrics=str(result_metrics),
        model_type=model_type,
        start_p=start_p,
        start_q=start_q,
        max_p=max_p,
        max_q=max_q,
        m=m,
        start_P=start_P,
        D=s,
        seasonal=seasonal,
    ), model_type
