from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from pandas import DataFrame


def plot_dataset(df: DataFrame,
                 title: str):
    """
    Plot the dataset using Plotly.

    Args:
        df (DataFrame): Input DataFrame.
        title (str): Title for the plot.
    """

    data = []
    value = go.Scatter(
        x=df.index,
        y=df.value,
        mode="markers",
        name="Concentration Ntot (mg/L)",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Concentration Ntot (mg/L)", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    pio.show(fig)


def plot_predictions(df_result: DataFrame):
    """
    Plot predictions and actual values using Plotly.

    Args:
        df_result (DataFrame): DataFrame containing 'value' (actual values) and 'prediction' (predicted values).
    """

    data = []

    value = go.Scatter(
        x=df_result.index,
        y=df_result.value,
        mode="markers",
        name="Measured values",
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    prediction = go.Scatter(
        x=df_result.index,
        y=df_result.prediction,
        mode="markers",
        # line={"dash": "dot"},
        name='predictions',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction)

    layout = dict(
        title="Predictions vs Actual Values for the dataset",
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Concentration Ntot (mg/L)", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    pio.show(fig)


def plot_losses(train_losses: List, val_losses: List, best_epoch: int, best_val_loss: float):
    """
    Plot predictions and actual values using Plotly.

    Args:
        train_losses (List): List containing training losses.
        val_losses (List): List containing validation losses
        best_epoch (int): Best epoch of the training process
        best_val_loss (float): Best validation loss of the training process
    """

    data = []

    tr_lss = go.Scatter(
        x=list(range(len(train_losses))),
        y=train_losses,
        mode="lines",
        name="Training",
        marker=dict(),
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(tr_lss)

    vl_lss = go.Scatter(
        x=list(range(len(val_losses))),
        y=val_losses,
        mode="lines",
        name='Validation',
        marker=dict(),
        opacity=0.8,
    )
    data.append(vl_lss)

    layout = dict(
        title="Training and Validation losses",
        xaxis=dict(title="Epoch", ticklen=5, zeroline=False),
        yaxis=dict(title="Loss", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)

    fig = go.Figure(fig)

    fig.add_annotation(x=int(best_epoch) - 1, y=float(best_val_loss),
                       text=f"Best validation loss: {best_val_loss}",
                       showarrow=True,
                       arrowhead=1)
    fig.show()

    # pio.show(fig)


def plot_residuals(residuals: np.array):
    """
    Plot given residuals in plotly.

    Args:
        residuals (List): List containing residual values.
    """

    df = pd.DataFrame({'y': residuals,
                       'x': list(range(len(residuals)))})

    fig = px.scatter(df, x='x', y='y', trendline="lowess")
    fig.show()


def plot_predictions_and_forecasting(df_result: DataFrame, df_forecasting: DataFrame):
    """
    Plot predictions, measured values, and forecast values from DataFrames using Plotly.

    Args:
        df_result (DataFrame): DataFrame containing measured values and predictions.
        df_forecasting (DataFrame): DataFrame containing forecasted values.

    Returns:
        None
    """
    data = []

    value = go.Scatter(
        x=df_result.index,
        y=df_result.value,
        mode="markers",
        name="Measured values",
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    prediction = go.Scatter(
        x=df_result.index,
        y=df_result.prediction,
        mode="markers",
        name='predictions',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction)

    forecast = go.Scatter(
        x=df_forecasting.index,
        y=df_forecasting.value,
        mode="markers",
        name='Forecast',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(forecast)

    layout = dict(
        title="Predictions, Actual Values of the dataset and Forecast Values",
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Concentration Ntot (mg/L)", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    pio.show(fig)


from plotly import graph_objects as go
from pandas import DataFrame


def plot_armima_forecasting(fitted_series: pd.Series, lower_series: pd.Series,
                            upper_series: pd.Series, data: DataFrame, model_type: str):
    """
    Plot ARIMA model forecasting results along with actual measured values and confidence intervals using Plotly.

    Args:
        fitted_series (pd.Series): Time series data of the ARIMA model's fitted values.
        lower_series (pd.Series): Time series data of the lower confidence interval.
        upper_series (pd.Series): Time series data of the upper confidence interval.
        data (DataFrame): DataFrame containing measured values.
        model_type (str): The type of ARIMA model used for forecasting.

    Returns:
        None
    """
    plot_data = []

    main_data = go.Scatter(
        x=data.index,
        y=data.value,
        mode="markers",
        name="Measured values",
        marker=dict()
    )
    plot_data.append(main_data)

    fitted = go.Scatter(
        x=fitted_series.index,
        y=fitted_series.values,
        mode="markers",
        name='Forecast',
        marker=dict(),
    )
    plot_data.append(fitted)

    layout = dict(
        title=f"{model_type} Forecast Values vs Actual Values of the Dataset",
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Concentration Ntot (mg/L)", ticklen=5, zeroline=False),
    )

    fig = dict(data=plot_data, layout=layout)

    fig = go.Figure(fig)

    fig.add_traces([
        go.Scatter(x=upper_series.index,
                   y=upper_series.values,
                   mode='lines',
                   showlegend=False),
        go.Scatter(x=lower_series.index,
                   y=lower_series.values,
                   mode='lines',
                   name='95% confidence interval',
                   fill='tonexty')
    ])

    fig.show()


def plot_armima_prediction(fitted_series: pd.Series, lower_series: pd.Series,
                           upper_series: pd.Series, data: DataFrame, model_type: str):
    """
    Plot ARIMA model prediction results along with actual measured values and confidence intervals using Plotly.

    Args:
        fitted_series (pd.Series): Time series data of the ARIMA model's fitted values.
        lower_series (pd.Series): Time series data of the lower confidence interval.
        upper_series (pd.Series): Time series data of the upper confidence interval.
        data (DataFrame): DataFrame containing measured values.
        model_type (str): The type of ARIMA model used for forecasting.

    Returns:
        None
    """

    plot_data = []

    main_data = go.Scatter(
        x=data.index,
        y=data.value,
        mode="markers",
        name="Measured values",
        marker=dict()
    )
    plot_data.append(main_data)

    fitted = go.Scatter(
        x=fitted_series.index,
        y=fitted_series.values,
        mode="markers",
        name='Forecast',
        marker=dict(),
    )
    plot_data.append(fitted)

    layout = dict(
        title=f"{model_type} Predictions vs Actual Values of the Dataset",
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title="Concentration Ntot (mg\L)", ticklen=5, zeroline=False),
    )

    fig = dict(data=plot_data, layout=layout)

    fig = go.Figure(fig)

    fig.add_traces([
        go.Scatter(x=upper_series.index,
                   y=upper_series.values,
                   mode='lines',
                   showlegend=False),
        go.Scatter(x=lower_series.index,
                   y=lower_series.values,
                   mode='lines',
                   name='95% confidence interval',
                   fill='tonexty')
    ])

    fig.show()
