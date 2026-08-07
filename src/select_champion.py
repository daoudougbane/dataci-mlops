import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = MlflowClient()

runs = mlflow.search_runs(experiment_names=['cacao_price_prediction'])
runs = runs.sort_values('start_time').drop_duplicates(
    subset=['tags.mlflow.runName'], keep='last'
)

cols = ['tags.mlflow.runName', 'metrics.rmse', 'metrics.mae', 'metrics.r2', 'metrics.latency_ms']
print(runs[cols].to_string(index=False))

# Grille de sélection : RMSE minimal ET latence < 50 ms
eligible = runs[runs['metrics.latency_ms'] < 50]
if eligible.empty:
    eligible = runs  # secours si aucun run ne respecte la contrainte de latence
champion_row = eligible.loc[eligible['metrics.rmse'].idxmin()]
run_id = champion_row['run_id']
champion_name = champion_row['tags.mlflow.runName']

print(f"\nChampion sélectionné : {champion_name} (run_id={run_id}, "
      f"RMSE={champion_row['metrics.rmse']:.2f}, latence={champion_row['metrics.latency_ms']:.2f}ms)")

# Enregistrer dans le Model Registry
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri, "cacao_price_predictor")

# Promouvoir en Staging
client.transition_model_version_stage(
    name="cacao_price_predictor",
    version=result.version,
    stage="Staging",
)

print(f"Champion en Staging : version {result.version}")

with open("reports/champion_selection.txt", "w") as f:
    f.write(runs[cols].to_string(index=False))
    f.write(f"\n\nChampion sélectionné : {champion_name} (run_id={run_id})\n")
    f.write(f"RMSE={champion_row['metrics.rmse']:.2f}  "
            f"MAE={champion_row['metrics.mae']:.2f}  "
            f"R2={champion_row['metrics.r2']:.4f}  "
            f"latence={champion_row['metrics.latency_ms']:.2f} ms\n")
    f.write(f"Model Registry : cacao_price_predictor, version {result.version}, stage=Staging\n")
