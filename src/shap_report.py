import pandas as pd
import mlflow
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mlflow.set_tracking_uri('sqlite:///mlflow.db')

FEATURES = ['lag_7', 'lag_30', 'rolling_mean_4', 'rolling_std_4', 'mois_sin', 'mois_cos']

test = pd.read_csv('data/features/test.csv', index_col='date', parse_dates=True)
X_te = test[FEATURES]

# Charger le modèle champion depuis le Registry (adapter selon l'algo : XGBoost)
model = mlflow.xgboost.load_model("models:/cacao_price_predictor/Staging")

# SHAP TreeExplainer (adapté aux modèles à base d'arbres : XGBoost/LightGBM)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_te)

# --- Summary Plot (global) ---
plt.figure()
shap.summary_plot(shap_values, X_te, show=False)
plt.tight_layout()
plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# Importance moyenne absolue par feature (pour le compte rendu)
import numpy as np
mean_abs = np.abs(shap_values.values).mean(axis=0)
importance = sorted(zip(FEATURES, mean_abs), key=lambda x: -x[1])
print("Importance des features (SHAP, |valeur| moyenne) :")
for feat, val in importance:
    print(f"  {feat}: {val:.2f}")

# --- Waterfall Plots (3 prédictions locales) ---
indices = [0, len(X_te)//2, len(X_te)-1]
for idx in indices:
    plt.figure()
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()
    plt.savefig(f'reports/shap_waterfall_{idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

with open('reports/shap_top_features.txt', 'w') as f:
    f.write("Top features (SHAP importance moyenne absolue) :\n")
    for feat, val in importance:
        f.write(f"  {feat}: {val:.2f}\n")

print("Rapport SHAP généré dans reports/")
