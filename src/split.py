"""
Stage DVC 'split' — Split chronologique 80% train / 20% test.

Important : le split est TEMPOREL (pas aléatoire). Pour une série
temporelle, un split aléatoire mélangerait passé et futur : le modèle
verrait des dates "futures" pendant l'entraînement et serait évalué sur
des dates "passées", ce qui produit une fuite d'information (data leakage)
et surestime la performance réelle en production, où seul le passé est
disponible pour prédire le futur.
"""
import pandas as pd
import sys


def split(input_path, train_out, test_out):
    df = pd.read_csv(input_path, index_col='date', parse_dates=True).sort_index()

    # Split temporel : 80% train / 20% test — jamais aléatoire !
    n = len(df)
    cut = int(n * 0.8)
    train = df.iloc[:cut]
    test = df.iloc[cut:]

    train.to_csv(train_out)
    test.to_csv(test_out)

    print(f"[split] train={train.shape} test={test.shape}")


if __name__ == '__main__':
    split(sys.argv[1], sys.argv[2], sys.argv[3])
