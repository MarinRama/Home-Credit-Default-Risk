import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ce code sera exécuté dans le conteneur qui possède la base.
READ_DATABASE = """
import json
import os
import sqlite3
from contextlib import closing

path = os.environ["MONITORING_DB_PATH"]

with closing(sqlite3.connect(
    f"file:{path}?mode=ro", uri=True
)) as connection:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT * FROM prediction_events"
    ).fetchall()

print(json.dumps([dict(row) for row in rows], allow_nan=False))
"""


def export_run(scenario, run_id, database, feature_order, output_dir):
    journal_path = (
        PROJECT_ROOT / "data" / "simulation" / "runs"
        / f"{scenario}_{run_id}.jsonl"
    )

    journal = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(journal) != 1000:
        raise ValueError(
            f"{scenario} : 1000 appels attendus, {len(journal)} trouvés."
        )

    request_ids = [event["request_id"] for event in journal]
    if None in request_ids or len(set(request_ids)) != len(request_ids):
        raise ValueError("Identifiants de requêtes absents ou dupliqués.")

    feature_rows = []
    metric_rows = []

    for call in journal:
        request_id = call["request_id"]

        if call["run_id"] != run_id or call["scenario"] != scenario:
            raise ValueError("Le journal mélange plusieurs exécutions.")

        if request_id not in database:
            raise ValueError(f"Événement absent de SQLite : {request_id}")

        event = database[request_id]

        if call["status_code"] != 200 or event["status_code"] != 200:
            raise ValueError(f"Appel non réussi : {request_id}")

        inputs = json.loads(event["inputs_json"])
        outputs = json.loads(event["outputs_json"])

        if set(inputs) != set(feature_order):
            raise ValueError(f"Schéma d'entrée incorrect : {request_id}")

        # Comparer le résultat stocké au résultat reçu par le client.
        client_output = call["response"]
        if not np.isclose(
            outputs["score"],
            client_output["score"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"Scores incohérents : {request_id}")

        for key in ["prediction", "threshold", "model_version"]:
            if outputs[key] != client_output[key]:
                raise ValueError(
                    f"Résultat incohérent pour {key} : {request_id}"
                )

        feature_rows.append(inputs)
        metric_rows.append({
            "request_id": request_id,
            "run_id": run_id,
            "scenario": scenario,
            "row_position": call["row_position"],
            "created_at": event["created_at"],
            "status_code": event["status_code"],
            "model_version": event["model_version"],
            "score": outputs["score"],
            "prediction": outputs["prediction"],
            "threshold": outputs["threshold"],
            "inference_ms": event["inference_ms"],
            "processing_ms": event["processing_ms"],
            "client_latency_ms": call["client_latency_ms"],
            "error_type": event["error_type"],
        })

    features = pd.DataFrame(feature_rows).loc[:, feature_order].astype(float)
    metrics = pd.DataFrame(metric_rows)

    # Même index pour relier les variables aux événements sans ambiguïté.
    features.index = pd.Index(request_ids, name="request_id")
    metrics = metrics.set_index("request_id", verify_integrity=True)

    if not features.index.equals(metrics.index):
        raise ValueError("Les variables et les métriques sont désalignées.")

    if np.isinf(features.to_numpy()).any():
        raise ValueError("Des variables contiennent des valeurs infinies.")

    timing_columns = ["inference_ms", "processing_ms", "client_latency_ms"]
    timings = metrics[timing_columns].to_numpy(dtype=float)
    if not np.isfinite(timings).all() or (timings < 0).any():
        raise ValueError("Des durées sont absentes ou invalides.")

    prefix = f"{scenario}_{run_id}"
    features_path = output_dir / f"{prefix}_features.parquet"
    events_path = output_dir / f"{prefix}_events.parquet"

    features.to_parquet(features_path)
    metrics.to_parquet(events_path)

    print(f"\nScénario : {scenario}")
    print(f"Événements retrouvés et vérifiés : {len(metrics)}/1000")
    print(f"Variables : {features.shape[1]}")
    print(f"Score moyen : {metrics['score'].mean():.4f}")
    print(
        "Part des prédictions de classe 1 : "
        f"{metrics['prediction'].mean():.1%}"
    )
    print(f"Variables exportées : {features_path}")
    print(f"Métriques exportées : {events_path}")

    return {
        "run_id": run_id,
        "count": len(metrics),
        "features": str(features_path.relative_to(PROJECT_ROOT)),
        "events": str(events_path.relative_to(PROJECT_ROOT)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-run", required=True)
    parser.add_argument("--drift-run", required=True)
    args = parser.parse_args()

    schema = json.loads(
        (PROJECT_ROOT / "models" / "feature_schema.json")
        .read_text(encoding="utf-8")
    )

    result = subprocess.run(
        [
            "docker", "compose", "exec", "-T",
            "api", "python", "-c", READ_DATABASE,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Lecture de SQLite impossible :\n{result.stderr}"
        )

    rows = json.loads(result.stdout)
    database = {row["request_id"]: row for row in rows}

    output_dir = PROJECT_ROOT / "data" / "production" / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for scenario, run_id in [
        ("stable", args.stable_run),
        ("drift", args.drift_run),
    ]:
        manifest[scenario] = export_run(
            scenario,
            run_id,
            database,
            schema["feature_order"],
            output_dir,
        )

    manifest_path = output_dir / "monitoring_exports.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nManifeste : {manifest_path}")


if __name__ == "__main__":
    main()