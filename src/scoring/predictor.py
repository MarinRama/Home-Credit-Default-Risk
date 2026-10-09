import json
import math
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Contraintes du contrat d'entrée.
# None signifie qu'aucune limite n'est définie de ce côté.
FEATURE_BOUNDS = {
    "AMT_CREDIT": (0, None),
    "AMT_INCOME_TOTAL": (0, None),
    "AMT_ANNUITY": (0, None),
    "AMT_GOODS_PRICE": (0, None),
    "CNT_CHILDREN": (0, None),
    "CNT_FAM_MEMBERS": (1, None),
    "OWN_CAR_AGE": (0, None),
    "AGE_YEARS": (0, None),
    "EXT_SOURCE_1": (0, 1),
    "EXT_SOURCE_2": (0, 1),
    "EXT_SOURCE_3": (0, 1),
}

INTEGER_FEATURES = {
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
}

class CreditPredictor:
    def __init__(self, models_dir=None):
        models_dir = (
            Path(models_dir)
            if models_dir is not None
            else PROJECT_ROOT / "models"
        )

        self.model = joblib.load(
            models_dir / "final_credit_model.joblib"
        )

        with (models_dir / "feature_schema.json").open(
            encoding="utf-8"
        ) as file:
            self.schema = json.load(file)

        with (models_dir / "threshold_config.json").open(
            encoding="utf-8"
        ) as file:
            config = json.load(file)

        with (models_dir / "model_metadata.json").open(
            encoding="utf-8"
        ) as file:
            self.metadata = json.load(file)

        self.feature_names = self.schema["feature_order"]
        self.expected_features = set(self.feature_names)
        self.threshold = float(config["optimal_threshold"])

        if self.feature_names != list(self.model.feature_name_):
            raise ValueError(
                "Le schéma ne correspond pas aux variables du modèle."
            )

        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Le seuil enregistré est invalide.")

        classes = list(self.model.classes_)

        if classes != [0, 1]:
            raise ValueError("Les classes attendues sont 0 et 1.")

        self.default_class_index = classes.index(1)

    def prepare_input(self, features):
        """Valide une observation et respecte l'ordre du modèle."""
        if not isinstance(features, dict):
            raise ValueError("Les variables doivent être un dictionnaire.")

        received_features = set(features)

        missing = self.expected_features - received_features
        unexpected = received_features - self.expected_features

        if missing or unexpected:
            raise ValueError(
                f"Variables absentes : {sorted(missing)} ; "
                f"variables inconnues : {sorted(unexpected, key=str)}"
            )

        values = []

        for name in self.feature_names:
            value = features[name]

            # None correspond à null dans une requête JSON.
            if value is None:
                values.append(np.nan)
                continue

            # Les indicateurs doivent être transmis comme 0 ou 1.
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(
                    f"{name} doit contenir un nombre ou null."
                )

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"{name} contient NaN ou une valeur infinie. "
                    "Utiliser null pour une valeur manquante."
                )

            bounds = FEATURE_BOUNDS.get(name)

            if bounds is not None:
                minimum, maximum = bounds

                if minimum is not None and numeric_value < minimum:
                    raise ValueError(
                        f"{name} doit être supérieur ou égal à {minimum}."
                    )

                if maximum is not None and numeric_value > maximum:
                    raise ValueError(
                        f"{name} doit être inférieur ou égal à {maximum}."
                    )

            if name in INTEGER_FEATURES and not numeric_value.is_integer():
                raise ValueError(
                    f"{name} doit être un nombre entier."
                )
            
            values.append(numeric_value)

        if all(math.isnan(value) for value in values):
            raise ValueError(
                "Une observation entièrement vide ne peut pas être scorée."
            )

        return pd.DataFrame(
            [values],
            columns=self.feature_names,
        )

    def predict(self, features):
        prepared = self.prepare_input(features)

        score = float(
            self.model.predict_proba(prepared)[
                0, self.default_class_index
            ]
        )

        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError("Le modèle a retourné un score invalide.")

        return {
            "score": score,
            "prediction": int(score >= self.threshold),
            "threshold": self.threshold,
            "model_version": self.metadata["model_version"],
        }
    