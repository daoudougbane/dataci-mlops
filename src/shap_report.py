import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mlflow.set_tracking_uri('sqlite:///mlflow.db')

FEATURES = ['lag_7', 'lag_30', 'rolling_mean_4', 'rolling_std_4', 'mois_sin', 'mois_cos']

test = pd.read_csv('data/features/test.csv', index_col='date', parse_dates=True)
X_te = test[FEATURES]

# Charger le modèle champion depuis le Registry
sk_model = mlflow.xgboost.load_model("models:/cacao_price_predictor/Staging")
booster = sk_model.get_booster()

# --- Calcul SHAP natif XGBoost ---
dmatrix = xgb.DMatrix(X_te, feature_names=FEATURES)
contribs = booster.predict(dmatrix, pred_contribs=True)  # (n, n_features + 1)
shap_vals_array = contribs[:, :-1]        # valeurs SHAP par feature
base_value = float(contribs[0, -1])       # valeur de base (bias), identique pour toutes les lignes

shap_values = shap.Explanation(
    values=shap_vals_array,
    base_values=np.full(len(X_te), base_value),
    data=X_te.values,
    feature_names=FEATURES,
)

# --- Summary Plot (global) ---
plt.figure()
shap.summary_plot(shap_values, X_te, show=False)
plt.tight_layout()
plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# Importance moyenne absolue par feature (pour le compte rendu)
mean_abs = np.abs(shap_vals_array).mean(axis=0)
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