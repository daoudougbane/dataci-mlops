"""
Stage DVC 'validate' — Suite Great Expectations (5 contraintes qualité)
sur les données nettoyées de prix du cacao.

Contraintes appliquées :
1. La colonne 'date' ne doit jamais être nulle (clé temporelle obligatoire).
2. 'prix_xof' doit être strictement positif (0 < prix <= 20000 XOF, borne
   large pour couvrir les pics de marché sans laisser passer les valeurs
   aberrantes comme les -250 XOF vues dans le brut).
3. 'humidite_pct' doit être compris entre 0 et 100 (c'est un pourcentage).
4. 'pays_origine' doit appartenir à un ensemble de pays producteurs connus
   (cohérence référentielle).
5. La colonne 'date' ne doit contenir aucune valeur dupliquée (unicité de
   la clé temporelle, condition nécessaire pour un split chronologique
   propre en aval).

Le script utilise l'API Fluent de Great Expectations 1.x (contexte
éphémère + Data Docs générés sur disque local).
"""
import sys
import json
import great_expectations as gx
from great_expectations.core.batch_definition import BatchDefinition
import pandas as pd


PAYS_CONNUS = [
    "Côte d'Ivoire", "Ghana", "Nigéria", "Cameroun", "Togo",
    "Équateur", "Indonésie", "Brésil",
]


def validate(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])

    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(name="cacao_source")
    data_asset = data_source.add_dataframe_asset(name="cacao_clean")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "cacao_batch"
    )

    suite = context.suites.add(
        gx.ExpectationSuite(name="cacao_price_quality_suite")
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="date")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="prix_xof", min_value=0.01, max_value=20000, strict_min=True
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="humidite_pct", min_value=0, max_value=100
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="pays_origine", value_set=PAYS_CONNUS
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="date")
    )

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="cacao_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    batch_parameters = {"dataframe": df}
    results = validation_definition.run(batch_parameters=batch_parameters)

    n_success = sum(1 for r in results["results"] if r["success"])
    n_total = len(results["results"])
    overall = results["success"]

    print(f"[validate] {n_success}/{n_total} expectations passées "
          f"— Validation {'OK' if overall else 'ECHEC'}")

    # --- Rapport HTML (Data Docs) ---
    import os
    import shutil

    context.build_data_docs()
    docs_sites = context.get_docs_sites_urls()

    # Copier le site généré (chemin temporaire éphémère) vers l'emplacement
    # attendu par le TP : gx/uncommitted/data_docs/local_site/
    docs_dir = os.path.abspath("gx/uncommitted/data_docs/local_site")
    if docs_sites:
        src_url = docs_sites[0]["site_url"]
        src_path = src_url.replace("file://", "")
        src_dir = os.path.dirname(src_path)
        if os.path.isdir(src_dir):
            os.makedirs(os.path.dirname(docs_dir), exist_ok=True)
            if os.path.isdir(docs_dir):
                shutil.rmtree(docs_dir)
            shutil.copytree(src_dir, docs_dir)
            docs_sites = [{"site_name": "local_site",
                            "site_url": f"file://{docs_dir}/index.html"}]

    summary = {
        "success": overall,
        "n_success": n_success,
        "n_total": n_total,
        "expectations": [
            {
                "type": r["expectation_config"]["type"],
                "column": r["expectation_config"]["kwargs"].get("column"),
                "success": r["success"],
            }
            for r in results["results"]
        ],
        "data_docs_urls": docs_sites,
    }
    with open("gx/validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if not overall:
        sys.exit(1)


if __name__ == '__main__':
    validate(sys.argv[1])
