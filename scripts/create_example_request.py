import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    sample = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed"
        / "api_test_sample.parquet"
    )

    with (
        PROJECT_ROOT / "models" / "feature_name_mapping.json"
    ).open(encoding="utf-8") as file:
        mapping = json.load(file)

    with (
        PROJECT_ROOT / "models" / "feature_schema.json"
    ).open(encoding="utf-8") as file:
        schema = json.load(file)

    row = (
        sample.iloc[[0]]
        .drop(columns=["SK_ID_CURR"])
        .rename(columns=mapping)
    )

    row = row.loc[:, schema["feature_order"]].astype(float)

    features = json.loads(
        row.to_json(orient="records", double_precision=15)
    )[0]

    destination = (
        PROJECT_ROOT / "data" / "simulation"
        / "example_request.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        json.dump(
            {"features": features},
            file,
            indent=2,
            allow_nan=False,
        )

    print(f"Requête créée : {destination}")


if __name__ == "__main__":
    main()