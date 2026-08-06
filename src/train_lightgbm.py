import pandas as pd
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('cacao_price_prediction')

TARGET = 'prix_xof'
FEATURES = ['lag_7', 'lag_30', 'rolling_mean_4', 'rolling_std_4', 'mois_sin', 'mois_cos']

train = pd.read_csv('data/features/train.csv', index_col='date', parse_dates=True)
test = pd.read_csv('data/features/test.csv', index_col='date', parse_dates=True)

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

params_lgb = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6,
              'num_leaves': 63, 'random_state': 42, 'verbose': -1}

with mlflow.start_run(run_name='LightGBM_v1') as run:
    mlflow.log_params(params_lgb)
    mlflow.set_tag('model_family', 'gradient_boosting')
    mlflow.set_tag('features', ','.join(FEATURES))

    model = lgb.LGBMRegressor(**params_lgb)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    rep = X_test.sample(1000, replace=True, random_state=42)
    t0 = time.perf_counter()
    model.predict(rep)
    latency_ms = (time.perf_counter() - t0) * 1000

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)
    mlflow.log_metric('latency_ms', latency_ms)

    mlflow.lightgbm.log_model(model, name='model')

    print(f"[LightGBM_v1] RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.4f} latency_ms={latency_ms:.2f}")
    print(f"run_id={run.info.run_id}")
