from src.data.load_data import load_data
from src.data.split_data import train_test_split_by_date, Xy_split
from src.features import preprocess
from src.train.baseline import train_model
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


df = load_data()
df = preprocess(df)
train, test = train_test_split_by_date(df)
X_train, y_train = Xy_split(train)
X_test, y_test = Xy_split(test)

params = {'n_estimators': 155,
                      'max_depth': 2,
                      'learning_rate': 0.19672000712560175,
                      'subsample': 0.8947515745740601,
                      'colsample_bytree': 0.8685448913893068,
                      'min_child_weight': 9,
                      'reg_alpha': 2.1674832038511567,
                      'reg_lambda': 0.4674721811076263}

model = XGBRegressor(**params, random_state=42)
model.fit(X_train, y_train)

y_preds = model.predict(X_test)

print("RMSE: ", root_mean_squared_error(y_test, y_preds))
print("MAE: ", mean_absolute_error(y_test, y_preds))
print("R2: ", r2_score(y_test, y_preds))