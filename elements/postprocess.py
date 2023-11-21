import pandas as pd
import numpy as np
from typing import List


def inverse_transform(scaler, df: pd.DataFrame, columns: List) -> pd.DataFrame:
    """
    Inverse transform scaled columns of a DataFrame using the specified scaler.

    Args:
        scaler: Scaler object for inverse transformation.
        df (DataFrame): Input DataFrame.
        columns (list): List of columns to inverse transform.

    Returns:
        DataFrame: DataFrame with specified columns inverse transformed.
    """
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions: List, values: List, df_test: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Format predictions and actual values into a DataFrame and inverse transform if necessary.

    Args:
        predictions (list): List of prediction arrays.
        values (list): List of actual value arrays.
        df_test (DataFrame): Test DataFrame for index.
        scaler: Scaler object for inverse transformation.

    Returns:
        DataFrame: DataFrame containing 'value' (actual values) and 'prediction' (predicted values).
    """
    vals = np.concatenate(values, axis=0).ravel()
    prd = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": prd}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result


def remove_outliers(df: pd.DataFrame, bias: int):
    """
    Remove rows from a DataFrame where the 'value' column is greater than or equal to a specified bias.

    Args:
        df (pandas.DataFrame): The input DataFrame containing data to filter.
        bias (float): The threshold value used to filter the 'value' column. Rows with 'value' greater than or equal
                     to this bias will be removed.

    Returns:
        pandas.DataFrame: A new DataFrame with rows removed based on the bias.
    """
    df = df[df['value'] < bias]
    return df
