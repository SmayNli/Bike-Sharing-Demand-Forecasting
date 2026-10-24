import pandas as pd
import yaml

def load_data(data_name = "day"):

    with open("configs.yaml", "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    if data_name == "day":
        data = pd.read_csv(configs["data"]["day_data_path"])
    elif data_name == "hour":
        data = pd.read_csv(configs["data"]["hour_data_path"])

    df = data.copy()

    return df

if __name__ == "__main__":
    df = load_data(data_name="hour")

    print(df.head())