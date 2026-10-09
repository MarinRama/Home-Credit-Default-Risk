import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring.predictor import CreditPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    predictor = CreditPredictor()

    sample = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed"
        / "api_test_sample.parquet"
    )

    with (
        PROJECT_ROOT / "models" / "feature_name_mapping.json"
    ).open(encoding="utf-8") as file:
        mapping = json.load(file)

    client_id = int(sample.iloc[0]["SK_ID_CURR"])

    # Préparer une ligne avec les noms utilisés par le modèle.
    row = (
        sample.iloc[[0]]
        .drop(columns=["SK_ID_CURR"])
        .rename(columns=mapping)
    )

    if set(row.columns) != set(predictor.feature_names):
        raise ValueError("Les colonnes de l'échantillon sont incorrectes.")

    row = row.loc[:, predictor.feature_names].astype(float)

    # to_json convertit les valeurs manquantes en null.
    features = json.loads(
        row.to_json(orient="records", double_precision=15)
    )[0]

    result = predictor.predict(features)

    direct_score = float(
        predictor.model.predict_proba(row)[0, 1]
    )

    np.testing.assert_allclose(
        result["score"],
        direct_score,
        rtol=0,
        atol=1e-10,
    )

    print(f"Client testé : {client_id}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Score identique à l'appel direct : OK")

    # Vérifier qu'une variable oubliée est bien détectée.
    invalid_features = features.copy()
    removed_feature = predictor.feature_names[0]
    invalid_features.pop(removed_feature)

    try:
        predictor.predict(invalid_features)
    except ValueError as error:
        print(f"Requête incomplète rejetée : OK ({error})")
    else:
        raise AssertionError(
            "La requête incomplète aurait dû être rejetée."
        )


if __name__ == "__main__":
    main()