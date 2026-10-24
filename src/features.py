from src.data.load_data import load_data
import pandas as pd

def clean_data(df: pd.DataFrame):
    df = df.drop(["temp", "instant"], axis=1)
    df["dteday"] = pd.to_datetime(df["dteday"])

    return df

def lag_features(df: pd.DataFrame):
    df["cnt_lag_1"] = df["cnt"].shift(1)
    df["cnt_lag_7"] = df["cnt"].shift(7)

    df["cnt_rolling7"] = df["cnt"].shift(1).rolling(window=7).mean()
    df["cnt_rolling30"] = df["cnt"].shift(1).rolling(window=30).mean()

    df = df.dropna(subset = "cnt_rolling30")

    return df

def preprocess(df: pd.DataFrame):
    df = clean_data(df)
    df = lag_features(df)

    return df

if __name__ == "__main__":
    df = load_data()

    df = preprocess(df)

    print(df.head())




