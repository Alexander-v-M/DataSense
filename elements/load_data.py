import pandas as pd


def get_data() -> pd.DataFrame:
    """
    Load and preprocess the dataset.

    Returns:
        DataFrame: Preprocessed DataFrame with datetime index and a renamed 'value' column.
    """
    # Load the data from the CSV file
    df = pd.read_csv(
        r'C:\Users\avanmeekeren\OneDrive - Wetterskip Fryslân\stage\main_project\DataSense\data\PJME_hourly.csv')

    # Set the 'Datetime' column as the index
    df = df.set_index(['Datetime'])
    df.index = pd.to_datetime(df.index)

    # Sort the DataFrame by date if it's not already sorted
    # if not df.index.is_monotonic:
    df = df.sort_index()

    # Rename the main column to 'value'
    df = df.rename(columns={'PJME_MW': 'value'})
    df = df.groupby(pd.Grouper(freq='1W')).mean()
    return df


def get_data_wf(dataset) -> pd.DataFrame:
    """
    Load and preprocess the dataset.

    Returns:
        DataFrame: Preprocessed DataFrame with datetime index and a renamed 'value' column.
    """
    # Load the data from the CSV file
    df = pd.read_csv(
        fr'C:\Users\avanmeekeren\OneDrive - Wetterskip Fryslân\stage\main_project\DataSense\data\ts{dataset}.csv',
        sep=';',
        encoding='latin-1',
        on_bad_lines='skip',
        skip_blank_lines=True, low_memory=False)

    # Set the 'Datetime' column as the index
    df = df.set_index(['Datum'])
    df.index = pd.to_datetime(df.index, dayfirst=True)

    # Sort the DataFrame by date if it's not already sorted
    # if not df.index.is_monotonic:
    df = df.sort_index()

    # Rename the main column to 'value'
    df = df.rename(columns={'Numeriek': 'value'})
    df = df[["value"]]
    df["value"] = df["value"].str.replace(',', '.').astype(float)
    df.dropna(inplace=True)
    df = df[["value"]]
    df = df.groupby(pd.Grouper(freq='1W')).mean()
    df.dropna(inplace=True)

    return df
