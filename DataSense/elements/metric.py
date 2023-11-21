import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score


# Calculate SMAPE
def smape(a, f):
    return 200 * np.mean(np.abs(a - f) / (np.abs(a) + np.abs(f)))


def mase(a, f, m=12):
    k = len(a)
    n = len(f)

    numerator = (1 / k) * np.sum(np.abs(a - f))
    denominator = (1 / (n - m)) * np.sum(np.abs(a - np.roll(a, -m)[:n]))

    mase_seasonal = numerator / denominator

    return mase_seasonal


def u(a, f):
    n = len(a)

    numerator = mean_squared_error(a, f) ** 0.5
    denominator = ((1 / n * np.sum(f) ** 2) ** 0.5) * ((1 / n * np.sum(a) ** 2) ** 0.5)

    return numerator / denominator


def calculate_metrics(df: DataFrame, lis=False):
    """
    Calculate regression metrics for evaluating model performance.

    Args:
        df (DataFrame): DataFrame containing 'value' (actual values) and 'prediction' (predicted values).
        lis (bool): return the metrics on List format

    Returns:
        dict: A dictionary of regression metrics including RMSE, R2, sMAPE, MASE, U-statistic.
    """

    if lis:
        return [mean_squared_error(df.value, df.prediction) ** 0.5, r2_score(df.value, df.prediction),
                smape(df.value, df.prediction), mase(df.value, df.prediction), u(df.value, df.prediction)]
    else:
        return (f"rmse: {mean_squared_error(df.value, df.prediction) ** 0.5}\n"
                f"r2:{r2_score(df.value, df.prediction)}\nsMAPE:{smape(df.value, df.prediction)}\n"
                f"MASE:{mase(df.value, df.prediction)}\nU:{u(df.value, df.prediction)}")


def calculate_metrics_no_df(fitted, actual, lis=False):
    """
    Calculate regression metrics for evaluating model performance.

    Args:
        fitted (series): Pandas Series containing predicted values
        actual (series): Pandas Series containing actual values
        lis (bool): return the metrics on List format

    Returns:
        dict: A dictionary of regression metrics including RMSE, R2, sMAPE, MASE, U-statistic.
    """

    actual = actual.value
    fitted = fitted.values

    if lis:
        return [mean_squared_error(actual, fitted) ** 0.5, r2_score(actual, fitted), smape(actual, fitted),
                mase(actual, fitted), u(actual, fitted)]
    else:

        return (f"rmse:{mean_squared_error(actual, fitted) ** 0.5}\n"
                f"r2:{r2_score(actual, fitted)}\nsMAPE:{smape(actual, fitted)}\n"
                f"MASE:{mase(actual, fitted)}\nU:{u(actual, fitted)}")


def calc_residuals(df):
    residuals = df.value - df.prediction

    m_res = np.mean(residuals)

    return residuals, m_res
