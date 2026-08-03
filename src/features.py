"""
Stage DVC 'features' — Feature engineering pour la prédiction du prix
du cacao à J+7.

Les données sont hebdomadaires (une ligne = une semaine), donc :
- lag_7  = shift(1)  -> prix de la semaine précédente (7 jours)
- lag_30 = shift(4)  -> prix il y a ~4 semaines (30 jours)
On garde ici les deux conventions du sujet ; voir compte-rendu (question 1)
pour la justification du choix shift(n) selon la fréquence des données.

Features créées (6) :
1. lag_7            : prix_xof décalé d'1 semaine
2. lag_30           : prix_xof décalé de 4 semaines (~30 jours)
3. rolling_mean_4   : moyenne mobile sur 4 semaines
4. rolling_std_4    : écart-type mobile sur 4 semaines
5. mois_sin         : encodage cyclique du mois (sinus)
6. mois_cos         : encodage cyclique du mois (cosinus)
"""
import pandas as pd
import numpy as np
import sys


def build_features(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=['date']).set_index('date').sort_index()

    # Combler les petits trous des variables explicatives secondaires
    df[['volume_tonnes', 'humidite_pct', 'qualite_index', 'temp_moy_c']] = (
        df[['volume_tonnes', 'humidite_pct', 'qualite_index', 'temp_moy_c']]
        .ffill()
    )

    # --- Features lag ---
    # Données hebdomadaires : 1 ligne = 7 jours -> shift(1) = lag de 7 jours
    df['lag_7'] = df['prix_xof'].shift(1)
    # Un lag de 30 jours (~1 mois) correspond à shift(4) car 4 semaines * 7j = 28j
    df['lag_30'] = df['prix_xof'].shift(4)

    # --- Features rolling ---
    df['rolling_mean_4'] = df['prix_xof'].shift(1).rolling(window=4).mean()
    df['rolling_std_4'] = df['prix_xof'].shift(1).rolling(window=4).std()

    # --- Encodage cyclique du mois ---
    mois = df.index.month
    df['mois_sin'] = np.sin(2 * np.pi * mois / 12)
    df['mois_cos'] = np.cos(2 * np.pi * mois / 12)

    # NB : la cible utilisée dans le TP2 (train_xgboost.py) est directement
    # la colonne 'prix_xof' (prix courant), prédite à partir des features
    # décalées (lag_7, lag_30, ...) qui ne contiennent aucune fuite du futur.

    # Supprimer les lignes incomplètes générées par les lags/rolling
    feature_cols = ['lag_7', 'lag_30', 'rolling_mean_4', 'rolling_std_4',
                     'mois_sin', 'mois_cos']
    df = df.dropna(subset=feature_cols + ['prix_xof'])

    df.to_csv(output_path)
    print(f"[features] {len(df)} lignes, {len(feature_cols)} features créées")


if __name__ == '__main__':
    build_features(sys.argv[1], sys.argv[2])
