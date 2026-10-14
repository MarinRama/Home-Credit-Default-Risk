import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_SEED = 42
N_REFERENCE = 5000
N_PRODUCTION = 1000


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    source = (
        PROJECT_ROOT / "data" / "processed"
        / "application_train_enriched.parquet"
    )

    mapping_path = (
        PROJECT_ROOT / "models" / "feature_name_mapping.json"
    )
    schema_path = PROJECT_ROOT / "models" / "feature_schema.json"

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_order = schema["feature_order"]

    data = pd.read_parquet(source)

    if not data.columns.is_unique:
        raise ValueError("Les noms de colonnes ne sont pas uniques.")

    if data["SK_ID_CURR"].isna().any():
        raise ValueError("Des identifiants clients sont manquants.")

    if data["SK_ID_CURR"].duplicated().any():
        raise ValueError("Des identifiants clients sont dupliqués.")

    required_rows = N_REFERENCE + N_PRODUCTION
    if len(data) < required_rows:
        raise ValueError("Nombre de clients insuffisant.")

    # Un seul tirage sans remise garantit deux ensembles disjoints.
    selected = data.sample(
        n=required_rows,
        replace=False,
        random_state=RANDOM_SEED,
    )

    reference_ids = (
        selected.iloc[:N_REFERENCE]["SK_ID_CURR"]
        .astype(int)
        .tolist()
    )
    production_ids = (
        selected.iloc[N_REFERENCE:]["SK_ID_CURR"]
        .astype(int)
        .tolist()
    )

    overlap = set(reference_ids) & set(production_ids)
    if overlap:
        raise ValueError("Des clients sont communs aux deux ensembles.")

    features = (
        selected
        .drop(columns=["SK_ID_CURR", "TARGET"])
        .rename(columns=mapping)
    )

    if not features.columns.is_unique:
        raise ValueError("Le renommage crée des colonnes dupliquées.")

    missing = sorted(set(feature_order) - set(features.columns))
    unexpected = sorted(set(features.columns) - set(feature_order))

    if missing or unexpected:
        raise ValueError(
            f"Variables absentes : {missing} ; "
            f"variables inconnues : {unexpected}"
        )

    features = features.loc[:, feature_order].astype(float)

    # Les valeurs manquantes sont conservées, mais pas les infinis.
    if np.isinf(features.to_numpy()).any():
        raise ValueError("Les données contiennent des valeurs infinies.")

    if features.isna().all(axis=1).any():
        raise ValueError("Un client possède uniquement des valeurs nulles.")

    reference = features.iloc[:N_REFERENCE].reset_index(drop=True)
    production = features.iloc[N_REFERENCE:].reset_index(drop=True)

    reference_dir = PROJECT_ROOT / "data" / "reference"
    simulation_dir = PROJECT_ROOT / "data" / "simulation"

    reference_dir.mkdir(parents=True, exist_ok=True)
    simulation_dir.mkdir(parents=True, exist_ok=True)

    reference_path = reference_dir / "reference_features.parquet"
    production_path = simulation_dir / "production_stable.parquet"

    reference.to_parquet(reference_path, index=False)
    production.to_parquet(production_path, index=False)

    # Identifiants conservés séparément pour la traçabilité locale.
    # Ils ne figurent pas parmi les variables envoyées au modèle.
    manifest = {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256_file(source),
        "schema_sha256": sha256_file(schema_path),
        "mapping_sha256": sha256_file(mapping_path),
        "random_seed": RANDOM_SEED,
        "n_features": len(feature_order),
        "n_reference": len(reference),
        "n_production_stable": len(production),
        "n_clients_in_common": len(overlap),
        "reference_client_ids": reference_ids,
        "production_client_ids": production_ids,
        "reference_sha256": sha256_file(reference_path),
        "production_stable_sha256": sha256_file(production_path),
        "purpose": "Simulation de monitoring, pas evaluation predictive",
    }

    manifest_path = simulation_dir / "monitoring_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"Référence : {len(reference)} clients, "
        f"{reference.shape[1]} variables"
    )
    print(
        f"Production stable : {len(production)} clients, "
        f"{production.shape[1]} variables"
    )
    print(f"Clients communs : {len(overlap)}")
    print("TARGET et SK_ID_CURR exclus des variables : OK")
    print("Ordre des variables conforme au contrat : OK")
    print(f"Référence enregistrée : {reference_path}")
    print(f"Production enregistrée : {production_path}")
    print(f"Manifeste enregistré : {manifest_path}")


if __name__ == "__main__":
    main()