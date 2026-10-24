from src.data.load_data import load_data
from src.features import preprocess
import pandas as pd

def train_test_split_by_date(df: pd.DataFrame, test_date = "2012-10-01"):

    train_data = df[df["dteday"] < test_date].copy()
    test_data = df[df["dteday"] >= test_date].copy()

    return train_data, test_data

def Xy_split(df: pd.DataFrame):
    X = df.drop(["dteday", "casual", "registered", "cnt"], axis = 1)
    y = df["cnt"]

    return X, y



if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    train_data, test_data = train_test_split_by_date(df = df)
    X, y = Xy_split(train_data)