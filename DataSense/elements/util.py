from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from pandas import DataFrame
from typing import Tuple


def feature_label_split(df: DataFrame, target_col: str) -> Tuple:
    """
    Split a DataFrame into feature matrix (X) and target vector (y).

    Args:
        df (DataFrame): Input DataFrame.
        target_col (str): Name of the target column.

    Returns:
        DataFrame, DataFrame: Feature matrix (X) and target vector (y).
    """
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y


def train_val_test_split(df: DataFrame, target_col: str, test_ratio: float, validation: bool = True) -> Tuple:
    """
    Split a DataFrame into training, validation, and test sets.

    Args:
        df (DataFrame): Input DataFrame.
        target_col (str): Name of the target column.
        test_ratio (float): Ratio of data to allocate for testing.
        validation (bool): True if there is the need for a validation dataset

    Returns:
        DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame:
            X_train, X_val, X_test, y_train, y_val, y_test.
    """
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)

    if validation:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test


def get_scaler(scaler: str):
    """
    Get a scaler object based on the specified scaler name.

    Args:
        scaler (str): Name of the scaler ('minmax', 'standard', 'maxabs', 'robust').

    Returns:
        Scaler: Scaler object.
    """
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()
