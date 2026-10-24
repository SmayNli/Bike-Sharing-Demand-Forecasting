from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import numpy as np
from src.data.load_data import load_data
from src.data.split_data import train_test_split_by_date, Xy_split
from src.features import preprocess

def train_model(X, y, model=XGBRegressor()):
    tscv = TimeSeriesSplit()
    rmse_scores = []
    mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X=X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_preds = model.predict(X_val)

        rmse_scores.append(root_mean_squared_error(y_val, y_preds))
        mae_scores.append(mean_absolute_error(y_val, y_preds))

    print("RMSE: ", np.mean(rmse_scores))
    print("MAE: ", np.mean(mae_scores))

def main():
    df = load_data()
    df = preprocess(df)
    train, test = train_test_split_by_date(df)
    X, y = Xy_split(train)
    train_model(X, y)

if __name__ == "__main__":
    main()