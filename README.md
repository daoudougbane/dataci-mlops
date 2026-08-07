# GBANE DAOUDOU ET KONE DOFONA 
# Projet DataCI — Prédiction du prix du cacao (TP1 + TP2 MLOps)

Ce dossier contient l'implémentation complète et EXÉCUTÉE des TP1 (pipeline DVC)
et TP2 (expérimentation MLflow) sur les données réelles fournies
(`prix_cacao_fao_2015_2024.csv`), testée de bout en bout.

## Ce qui a déjà été fait et vérifié ici
- Pipeline DVC (`dvc repro`) : clean → validate → features → split, exécuté
  sans erreur (voir `dvc.yaml` / `dvc.lock`).
- Validation Great Expectations : 5/5 contraintes passées
  (`gx/uncommitted/data_docs/local_site/index.html`).
- 3 modèles entraînés et tracés dans MLflow (`mlflow.db`) : XGBoost, LightGBM, Prophet.
- Champion sélectionné (XGBoost) et promu en Staging dans le Model Registry.
- Rapport SHAP généré (`reports/`).
- Comptes rendus rédigés : `TP1_compte_rendu.docx`, `TP2_compte_rendu.docx`.

## Bug corrigé (important si tu relances toi-même)
`src/train_xgboost.py` fixe explicitement `base_score` (moyenne du prix sur
le train) dans les paramètres XGBoost. Sans ce réglage, XGBoost >= 3.0
auto-estime `base_score` et le sérialise sous forme de liste
(`"[1972.3]"`), ce que `shap.TreeExplainer` ne sait pas encore parser
(`ValueError: could not convert string to float`). Ce correctif règle le
problème à la racine — pas besoin d'y toucher si tu relances les scripts
tels quels.

## Ce qu'il TE reste à faire (ne peut pas être fait depuis ce sandbox)
1. **Créer le dépôt GitHub public `dataci-mlops`** et y pousser ce dossier
   (`git remote add origin <url> && git push -u origin master --tags`).
   Le dépôt git local est déjà initialisé avec l'historique et le tag `v1.0`.
2. **Screenshots réels** demandés par l'énoncé (dvc dag dans ton terminal,
   UI MLflow dans ton navigateur à `http://localhost:5000`, Model Registry) :
   remplacés ici par des visuels générés à partir des vraies données
   (`reports/dvc_dag.png`, `reports/mlflow_comparison.png`). Tu as déjà
   vérifié toi-même l'UI MLflow en local — si ton prof exige une capture
   d'écran, prends-la directement depuis ton navigateur.

## Comment relancer le pipeline toi-même
```bash
pip install dvc mlflow xgboost lightgbm prophet shap great-expectations
dvc repro                          # TP1 : pipeline de données
python src/train_xgboost.py        # TP2 : entraînement + tracking
python src/train_lightgbm.py
python src/train_prophet.py
python src/select_champion.py      # sélection + Model Registry
python src/shap_report.py          # rapport SHAP
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # UI web
```

## Structure
```
dataci-mlops/
├── data/{raw,clean,features}/   # données à chaque étape du pipeline
├── src/                         # tous les scripts (clean, validate, features,
│                                   split, train_*, select_champion, shap_report)
├── metrics/clean_metrics.json
├── gx/uncommitted/data_docs/    # rapport HTML Great Expectations
├── reports/                     # visuels + résultats (DAG, SHAP, comparaison)
├── mlflow.db                    # base de tracking MLflow (tous les runs)
├── dvc.yaml / dvc.lock          # définition du pipeline DVC
├── TP1_compte_rendu.docx
└── TP2_compte_rendu.docx
```
