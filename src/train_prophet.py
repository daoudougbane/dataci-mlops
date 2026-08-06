import pandas as pd
import mlflow
import mlflow.pyfunc
import time
import numpy as np
import logging
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('cacao_price_prediction')

train = pd.read_csv('data/features/train.csv', index_col='date', parse_dates=True)
test = pd.read_csv('data/features/test.csv', index_col='date', parse_dates=True)

# Prophet attend des colonnes 'ds' (date) et 'y' (valeur)
train_p = train.reset_index()[['date', 'prix_xof']].rename(columns={'date': 'ds', 'prix_xof': 'y'})
test_p = test.reset_index()[['date', 'prix_xof']].rename(columns={'date': 'ds', 'prix_xof': 'y'})


class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        forecast = self.model.predict(model_input)
        return forecast['yhat'].values


params_prophet = {'weekly_seasonality': True, 'yearly_seasonality': True,
                   'daily_seasonality': False, 'changepoint_prior_scale': 0.05}

with mlflow.start_run(run_name='Prophet_v1') as run:
    mlflow.log_params(params_prophet)
    mlflow.set_tag('model_family', 'time_series')

    model = Prophet(**params_prophet)
    model.fit(train_p)

    forecast = model.predict(test_p[['ds']])
    y_pred = forecast['yhat'].values
    y_test = test_p['y'].values

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    rep = test_p[['ds']].sample(1000, replace=True, random_state=42)
    t0 = time.perf_counter()
    model.predict(rep)
    latency_ms = (time.perf_counter() - t0) * 1000

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)
    mlflow.log_metric('latency_ms', latency_ms)

    mlflow.pyfunc.log_model(name='model', python_model=ProphetWrapper(model))

    print(f"[Prophet_v1] RMSE={rmse:.2f} MAE={mae:.2f} R2={r2:.4f} latency_ms={latency_ms:.2f}")
    print(f"run_id={run.info.run_id}")
