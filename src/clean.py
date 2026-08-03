"""
Stage DVC 'clean' — Nettoyage des données brutes FAO GIEWS (prix du cacao).

Règles appliquées :
1. Suppression des doublons sur la colonne 'date' (on garde la première occurrence).
2. Suppression des lignes avec prix_xof <= 0 (valeurs aberrantes/impossibles
   pour un prix de marché, ex: -250 XOF observé dans le brut).
3. Suppression des lignes où prix_xof est manquant (NaN) — une ligne sans
   la cible principale n'est pas exploitable pour l'entraînement.
"""
import pandas as pd
import sys
import json


def clean(input_path, output_path, metrics_path):
    df = pd.read_csv(input_path, parse_dates=['date'])
    n_before = len(df)

    # 1. Supprimer les doublons sur la colonne date
    df = df.drop_duplicates(subset=['date'], keep='first')

    # Supprimer les lignes où le prix cible est manquant (NaN)
    df = df.dropna(subset=['prix_xof'])

    # 2. Supprimer les lignes avec prix <= 0
    df = df[df['prix_xof'] > 0]

    df = df.sort_values('date').reset_index(drop=True)

    n_after = len(df)
    n_dropped = n_before - n_after

    df.to_csv(output_path, index=False)

    metrics = {
        "rows_in": n_before,
        "rows_out": n_after,
        "dropped": n_dropped,
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[clean] rows_in={n_before} rows_out={n_after} dropped={n_dropped}")


if __name__ == '__main__':
    clean(sys.argv[1], sys.argv[2], sys.argv[3])
