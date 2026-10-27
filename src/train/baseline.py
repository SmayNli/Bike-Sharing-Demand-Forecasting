from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from src.data.load_data import load_data
from src.data.split_data import train_test_split_by_date, Xy_split
from src.features import preprocess
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import pandas as pd

def train_model(X, y, model=XGBRegressor()):
    tscv = TimeSeriesSplit()
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X=X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_preds = model.predict(X_val)

        rmse_scores.append(root_mean_squared_error(y_val, y_preds))
        mae_scores.append(mean_absolute_error(y_val, y_preds))
        r2_scores.append(r2_score(y_val, y_preds))

    print("RMSE: ", np.mean(rmse_scores))
    print("MAE: ", np.mean(mae_scores))
    print("R2: ", np.mean(r2_scores), "\n")

    return np.mean(rmse_scores), np.mean(mae_scores), np.mean(r2_scores)

def run_experiment(X, y):
    params = {'n_estimators': 155,
                      'max_depth': 2,
                      'learning_rate': 0.19672000712560175,
                      'subsample': 0.8947515745740601,
                      'colsample_bytree': 0.8685448913893068,
                      'min_child_weight': 9,
                      'reg_alpha': 2.1674832038511567,
                      'reg_lambda': 0.4674721811076263}
    results = []
    models = {
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(**params, random_state=42),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1)
    }

    for model in models.items():
        print(model[0])
        rmse, mae, r2 = train_model(X, y, model[1])
        results.append({"name": model[0], "rmse": rmse, "mae": mae, "r2": r2})

    print(pd.DataFrame(results))

def main(run_experiment_ = False):
    df_day = load_data()
    df_hour = load_data("hour")
    df = preprocess(df_day, df_hour)
    train, test = train_test_split_by_date(df)
    X, y = Xy_split(train)

    if run_experiment_ == True:
        run_experiment(X, y)
    else:
        train_model(X, y)

if __name__ == "__main__":
    main()