import numpy as np
import pandas as pd


def calculate_ewma(df, columns):
    """
    Calculates the exponentially weighted moving average (EWMA) for the specified columns.
    """

    for col in columns:
        if col in df.columns:
            df[f"ewma15_{col}"] = df[col].ewm(span=15).mean()
    
    df= df.dropna(subset=[f"ewma15_{col}" for col in columns])
    
    return df

def calculate_rolling_mean(df, columns):
    """
    Calculates the rolling mean for the specified columns.
    """

    for col in columns:
        if col in df.columns:
            df[f"ema15_{col}"] = df[col].rolling(15).mean()
    
    df =df.dropna(subset=[f"ema15_{col}" for col in columns])
    
    return df

def messzeit_utc_handling(df, handling):
    """
    Cyclical encoding of the 'messzeit_utc' column into month, hour, and day-of-year features based on the specified handling options.
    """
    
    if handling.get("extract_month", False):
        if "messzeit_utc" in df.columns:
            # Month encoding (1-12 -> cyclical)
            df["month_sin"] = np.sin(2 * np.pi * pd.to_datetime(df["messzeit_utc"]).dt.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * pd.to_datetime(df["messzeit_utc"]).dt.month / 12)
            #df["month"] = pd.to_datetime(df["messzeit_utc"]).dt.month

    if handling.get("extract_time", False):
        if "messzeit_utc" in df.columns:
            # Hour encoding (0-23 -> cyclical)
            df["hour_sin"] = np.sin(2 * np.pi * pd.to_datetime(df["messzeit_utc"]).dt.hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * pd.to_datetime(df["messzeit_utc"]).dt.hour / 24)
            #df["hour"] = pd.to_datetime(df["messzeit_utc"]).dt.hour
        
    if handling.get("extract_doy", False):
        if "messzeit_utc" in df.columns:
            doy = pd.to_datetime(df["messzeit_utc"]).dt.dayofyear
            angle = (doy + 11) % 365 / 365 * 2 * np.pi 
            df["doy_cos"] = (-np.cos(angle) + 1) / 2
    return df

def transform_categorical_features(df):
    if "GEL_FORM" in df.columns:
        # Geländeform to binary in {0, 1}
        df['GEL_FORM'] = df['GEL_FORM'].apply(lambda x: 1 if x == 2 else 0)
    return df