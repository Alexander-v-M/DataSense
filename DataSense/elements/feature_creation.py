import pandas as pd
import numpy as np


def generate_time_lags(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """
    Generate time lag features in a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.
        n_lags (int): The number of lag features to generate.

    Returns:
        DataFrame: A DataFrame with time lag features.
    """
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n


def assign_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign additional time-related features to a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: A DataFrame with additional time-related features.
    """
    df_features = (
        df
        .assign(hour=df.index.hour)
        .assign(day=df.index.day)
        .assign(month=df.index.month)
        .assign(day_of_week=df.index.dayofweek)
        .assign(week_of_year=df.index.week)
    )
    return df_features


def onehot_encode_pd(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    One-hot encode a column in a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.
        col_name (str): The name of the column to one-hot encode.

    Returns:
        DataFrame: A DataFrame with one-hot encoded columns.
    """
    dummies = pd.get_dummies(df[col_name], prefix=col_name)
    return pd.concat([df, dummies], axis=1)


def generate_cyclical_features(df: pd.DataFrame,
                               col_name: str,
                               period: int,
                               start_num: int = 0) -> pd.DataFrame:
    """
    Generate cyclical features from a column in a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.
        col_name (str): The name of the column to generate cyclical features for.
        period (int): The period of the cyclical feature.
        start_num (int, optional): The starting point for the cyclical feature (default is 0).

    Returns:
        DataFrame: A DataFrame with cyclical features added.
    """
    kwargs = {
        f'sin_{col_name}': lambda x: np.sin(2 * np.pi * (df[col_name] - start_num) / period),
        f'cos_{col_name}': lambda x: np.cos(2 * np.pi * (df[col_name] - start_num) / period)
    }
    return df.assign(**kwargs).drop(columns=[col_name])


def get_features(df: pd.DataFrame, version: str, n_lag: int = None) -> pd.DataFrame:
    """
    Generate and extract features from a DataFrame based on the specified version.

    Args:
        df (DataFrame): The input DataFrame.
        version (str): The version of features to generate ('generated' or 'assigned').
        n_lag (int, optional): The number of lag features to generate for 'generated' version.

    Returns:
        DataFrame: A DataFrame with the selected features.
    """
    if version == 'generated':
        if n_lag is not None:
            df_features = generate_time_lags(df, n_lags=n_lag)
        else:
            raise ValueError(f"n_lag must be an integer, but it is {type(n_lag)}")

    elif version == 'assigned':
        df_features = assign_feature(df)
        df_features = onehot_encode_pd(df_features, 'month')
        df_features = onehot_encode_pd(df_features, 'day')
        df_features = onehot_encode_pd(df_features, 'day_of_week')
        df_features = onehot_encode_pd(df_features, 'week_of_year')
        df_features = generate_cyclical_features(df_features, 'hour', 24, 0)
        df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)
        df_features = generate_cyclical_features(df_features, 'month', 12, 1)
        df_features = generate_cyclical_features(df_features, 'week_of_year', 52, 0)

    else:
        raise NameError("Option for the feature parameter should be 'generated' or 'assigned'")

    return df_features
