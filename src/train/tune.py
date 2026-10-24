import optuna
from src.data.load_data import load_data
from src.data.split_data import train_test_split_by_date, Xy_split
from src.features import preprocess
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def objective(trial):

    params={
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 1, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
    }

    model = XGBRegressor(**params, random_state=42)

    scores = []
    tscv = TimeSeriesSplit()
    for train_idx, test_idx in tscv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_preds = model.predict(X_val)

        scores.append(root_mean_squared_error(y_val, y_preds))

    return np.mean(scores)

df = load_data()
df = preprocess(df)
train, test = train_test_split_by_date(df)
X, y = Xy_split(train)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Best RMSE: ", study.best_value)
print("Best Hyperparameters: ", study.best_params)