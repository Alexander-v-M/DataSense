import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

from DataSense.elements.metric import calculate_metrics_no_df

from DataSense.elements.util import train_val_test_split
from sklearn.linear_model import LinearRegression


def baseline(data):
    """
    Train and evaluate a baseline Linear Regression model for time series forecasting.

    Args:
        data (DataFrame): The time series dataset for training and evaluation.

    Returns:
        dict: A dictionary containing evaluation metrics for the baseline model.
    """
    # Split the data into training and testing sets
    _, _, y_train, y_test = train_val_test_split(data, validation=False, test_ratio=0.2, target_col='value')

    # Create a Linear Regression model
    model = LinearRegression()

    # Convert date index to ordinal for training
    y_train['date_ordinal'] = y_train.index
    y_train['date_ordinal'] = y_train['date_ordinal'].apply(lambda x: x.toordinal())

    # Fit the model to the training data
    model.fit(y_train['date_ordinal'].values.reshape(-1, 1), y_train['value'].values)

    # Convert date index to ordinal for testing
    y_test['date_ordinal'] = y_test.index
    y_test['date_ordinal'] = y_test['date_ordinal'].apply(lambda x: x.toordinal())

    # Make predictions using the model
    y_predict = model.predict(y_test['date_ordinal'].values.reshape(-1, 1))
    y_predict = pd.DataFrame(y_predict, index=y_test.index, columns=['value'])

    # Uncomment the following lines to visualize the predictions and actual values using matplotlib
    # plt.scatter(y_test.index, y_test.value)
    # plt.scatter(y_predict.index, y_predict.value)
    # plt.show()

    # Calculate evaluation metrics and return them in a dictionary
    return calculate_metrics_no_df(fitted=y_predict, actual=y_test, lis=True)

