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

def hourly_features(df: pd.DataFrame):
    bad_weather = list(df[(df["hr"]>=17) & (df["hr"]<=19) & (df["weathersit"].isin([3,4]))].index)
    df.loc[bad_weather, "rush_hour_bad_weather"] = int(1)
    df["rush_hour_bad_weather"] = df["rush_hour_bad_weather"].fillna(int(0))

    rush_hour_bad_weather = df.groupby("dteday")["rush_hour_bad_weather"].max()

    hourly_features = rush_hour_bad_weather

    return hourly_features

def preprocess(df_day: pd.DataFrame, df_hour: pd.DataFrame):
    df_day = clean_data(df_day)
    df_hour = clean_data(df_hour)

    df_day = lag_features(df_day)
    df_day = df_day.merge(hourly_features(df_hour), how="left", on="dteday")

    return df_day

if __name__ == "__main__":
    df_day = load_data()
    df_hour = load_data("hour")

    df = preprocess(df_day, df_hour)

    print(df.describe())




