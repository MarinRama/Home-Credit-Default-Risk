import csv
import hashlib
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "final_credit_model.joblib"
METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "features_metadata.csv"
)
THRESHOLD_PATH = MODELS_DIR / "threshold_config.json"


def clean_feature_names(columns):
    """Reproduit le nettoyage utilisé dans le notebook d'entraînement."""
    cleaned_columns = []
    name_counts = {}

    for column in columns:
        clean_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(column))
        clean_name = re.sub(r"_+", "_", clean_name).strip("_")

        if not clean_name:
            clean_name = "feature"

        if clean_name in name_counts:
            name_counts[clean_name] += 1
            clean_name = f"{clean_name}_{name_counts[clean_name]}"
        else:
            name_counts[clean_name] = 0

        cleaned_columns.append(clean_name)

    return cleaned_columns


def save_json(path, content):
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            content,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def main():
    # Vérifier les fichiers nécessaires avant de commencer.
    for path in (MODEL_PATH, METADATA_PATH, THRESHOLD_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    model = joblib.load(MODEL_PATH)

    with METADATA_PATH.open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    # L'identifiant et la cible ne sont pas des entrées du modèle.
    feature_rows = [
        row for row in rows
        if row["column"] not in {"SK_ID_CURR", "TARGET"}
    ]

    original_names = [row["column"] for row in feature_rows]
    cleaned_names = clean_feature_names(original_names)
    model_names = list(model.feature_name_)

    # Arrêter l'export si le contrat ne correspond pas au modèle.
    if len(set(original_names)) != len(original_names):
        raise ValueError("Les métadonnées contiennent des noms dupliqués.")

    if len(set(cleaned_names)) != len(cleaned_names):
        raise ValueError("Le nettoyage produit des noms dupliqués.")

    if cleaned_names != model_names:
        raise ValueError(
            "Les noms ou l'ordre des variables ne correspondent "
            "pas au modèle. Aucun export effectué."
        )

    if len(model_names) != model.n_features_in_:
        raise ValueError("Nombre de variables incohérent.")

    with THRESHOLD_PATH.open(encoding="utf-8") as file:
        threshold_config = json.load(file)

    threshold = float(threshold_config["optimal_threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Le seuil doit être compris entre 0 et 1.")

    mapping = dict(zip(original_names, cleaned_names))

    schema = {
        "schema_version": "1.0.0",
        "input_stage": "prepared_features",
        "n_features": len(model_names),
        "feature_order": model_names,
        "features": [
            {
                "name": cleaned_name,
                "original_name": row["column"],
                "source_dtype": row["dtype"],
            }
            for row, cleaned_name in zip(feature_rows, cleaned_names)
        ],
    }

    metadata = {
        "model_version": "1.0.0",
        "model_type": type(model).__name__,
        "model_file": MODEL_PATH.name,
        "model_sha256": hashlib.sha256(
            MODEL_PATH.read_bytes()
        ).hexdigest(),
        "n_features": int(model.n_features_in_),
        "classes": [int(value) for value in model.classes_],
        "decision_rule": "score >= threshold",
        "threshold_config": threshold_config,
        "export_environment": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in (
                    "lightgbm",
                    "scikit-learn",
                    "numpy",
                    "pandas",
                    "joblib",
                )
            },
        },
    }

    save_json(MODELS_DIR / "feature_name_mapping.json", mapping)
    save_json(MODELS_DIR / "feature_schema.json", schema)
    save_json(MODELS_DIR / "model_metadata.json", metadata)

    renamed_count = sum(
        original != cleaned
        for original, cleaned in zip(original_names, cleaned_names)
    )

    print(f"Modèle : {metadata['model_type']}")
    print(f"Variables vérifiées : {len(model_names)}")
    print(f"Variables renommées : {renamed_count}")
    print(f"Seuil conservé : {threshold}")
    print("Correspondance exacte des noms et de l'ordre : OK")
    print(f"Les trois fichiers JSON ont été créés dans : {MODELS_DIR}")


if __name__ == "__main__":
    main()