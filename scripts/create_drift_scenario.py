import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 43


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    simulation_dir = PROJECT_ROOT / "data" / "simulation"
    source = simulation_dir / "production_stable.parquet"
    destination = simulation_dir / "production_drift.parquet"
    report_path = simulation_dir / "drift_scenario.json"

    stable = pd.read_parquet(source)

    required = ["EXT_SOURCE_2", "EXT_SOURCE_3"]
    missing = [name for name in required if name not in stable.columns]
    if missing:
        raise ValueError(f"Variables absentes : {missing}")

    if stable.empty:
        raise ValueError("Le jeu de production stable est vide.")

    for name in required:
        values = stable[name].dropna()
        if values.empty:
            raise ValueError(f"{name} ne contient aucune valeur renseignée.")
        if not values.between(0, 1).all():
            raise ValueError(f"{name} contient des valeurs hors de [0, 1].")

    drift = stable.copy(deep=True)

    # Modification de la distribution, sans sortir de [0, 1].
    drift["EXT_SOURCE_2"] = stable["EXT_SOURCE_2"] * 0.60

    # Ajout de valeurs manquantes parmi les valeurs renseignées.
    rng = np.random.default_rng(RANDOM_SEED)
    eligible_positions = np.flatnonzero(
        stable["EXT_SOURCE_3"].notna().to_numpy()
    )
    n_removed = int(len(eligible_positions) * 0.20)

    if n_removed == 0:
        raise ValueError("Pas assez de valeurs pour simuler les suppressions.")

    removed_positions = rng.choice(
        eligible_positions,
        size=n_removed,
        replace=False,
    )
    column_position = drift.columns.get_loc("EXT_SOURCE_3")
    drift.iloc[removed_positions, column_position] = np.nan

    # Vérifier que seules les deux variables ciblées ont changé.
    other_columns = stable.columns.difference(required)
    pd.testing.assert_frame_equal(
        stable.loc[:, other_columns],
        drift.loc[:, other_columns],
    )

    if not drift.columns.equals(stable.columns):
        raise ValueError("L'ordre des variables a changé.")

    if len(drift) != len(stable):
        raise ValueError("Le nombre de clients a changé.")

    if np.isinf(drift.to_numpy(dtype=float)).any():
        raise ValueError("Le scénario contient des valeurs infinies.")

    summary = {}
    for name in required:
        summary[name] = {
            "mean_before": float(stable[name].mean()),
            "mean_after": float(drift[name].mean()),
            "missing_rate_before": float(stable[name].isna().mean()),
            "missing_rate_after": float(drift[name].isna().mean()),
        }

    drift.to_parquet(destination, index=False)

    report = {
        "scenario": "controlled_drift",
        "random_seed": RANDOM_SEED,
        "n_clients": len(drift),
        "source": str(source.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(destination),
        "changes": {
            "EXT_SOURCE_2": {"multiplication_factor": 0.60},
            "EXT_SOURCE_3": {
                "fraction_removed_among_observed": 0.20,
                "n_values_removed": n_removed,
            },
        },
        "summary": summary,
        "limitations": (
            "Test artificiel du monitoring. Les autres variables, "
            "y compris les variables derivees, ne sont pas recalculees. "
            "Les clients sont les memes que dans le scenario stable."
        ),
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Scénario créé : {len(drift)} clients")
    print("Ordre des variables et nombre de clients conservés : OK")
    print("Autres variables inchangées : OK")

    for name, metrics in summary.items():
        print(f"\n{name}")
        print(
            f"  Moyenne : {metrics['mean_before']:.4f}"
            f" → {metrics['mean_after']:.4f}"
        )
        print(
            f"  Valeurs manquantes : {metrics['missing_rate_before']:.1%}"
            f" → {metrics['missing_rate_after']:.1%}"
        )

    print(f"\nDonnées : {destination}")
    print(f"Description du scénario : {report_path}")


if __name__ == "__main__":
    main()