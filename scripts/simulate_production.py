import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["stable", "drift"],
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8001",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit doit être supérieur à zéro.")

    base_url = args.api_url.rstrip("/")
    source = (
        PROJECT_ROOT / "data" / "simulation"
        / f"production_{args.scenario}.parquet"
    )

    data = pd.read_parquet(source)
    if args.limit > len(data):
        parser.error(
            f"--limit dépasse le nombre de clients disponibles : {len(data)}"
        )

    data = data.iloc[:args.limit]

    # Vérifier l'API avant de commencer les envois.
    with urlopen(f"{base_url}/health", timeout=10) as response:
        health = json.load(response)

    if health.get("model_loaded") is not True:
        raise RuntimeError("Le modèle n'est pas chargé.")

    # Conversion des valeurs manquantes en null JSON.
    records = json.loads(
        data.to_json(orient="records", double_precision=15)
    )

    run_id = str(uuid.uuid4())
    output_dir = PROJECT_ROOT / "data" / "simulation" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.scenario}_{run_id}"
    journal_path = output_dir / f"{prefix}.jsonl"
    summary_path = output_dir / f"{prefix}_summary.json"

    results = []
    started_at = utc_now()

    print(f"Scénario : {args.scenario}")
    print(f"Exécution : {run_id}")
    print(f"Nombre de requêtes prévu : {len(records)}")

    with journal_path.open("w", encoding="utf-8") as journal:
        for position, features in enumerate(records):
            payload = json.dumps(
                {"features": features},
                allow_nan=False,
            ).encode("utf-8")

            request = Request(
                f"{base_url}/predict",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            event = {
                "run_id": run_id,
                "scenario": args.scenario,
                "row_position": position,
                "sent_at": utc_now(),
                "status_code": None,
                "request_id": None,
                "monitoring_status": None,
                "response": None,
                "transport_error": None,
            }

            start = time.perf_counter()

            try:
                try:
                    response = urlopen(request, timeout=30)
                except HTTPError as error:
                    # Une erreur HTTP reste une réponse à journaliser.
                    response = error

                with response:
                    raw_body = response.read().decode("utf-8")
                    event["status_code"] = response.code
                    event["request_id"] = response.headers.get(
                        "X-Request-ID"
                    )
                    event["monitoring_status"] = response.headers.get(
                        "X-Monitoring-Status"
                    )

                try:
                    event["response"] = json.loads(raw_body)
                except json.JSONDecodeError:
                    event["response"] = {"raw_body": raw_body}

            except (URLError, TimeoutError, OSError) as error:
                event["transport_error"] = str(error)

            event["client_latency_ms"] = (
                time.perf_counter() - start
            ) * 1000

            journal.write(
                json.dumps(event, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
            journal.flush()
            results.append(event)

            if (position + 1) % 100 == 0:
                print(f"{position + 1}/{len(records)} requêtes envoyées")

            if event["transport_error"]:
                print("Erreur réseau : arrêt des envois.")
                break

    successful = sum(
        event["status_code"] == 200 for event in results
    )
    stored = sum(
        event["monitoring_status"] == "stored" for event in results
    )
    request_ids = [
        event["request_id"]
        for event in results
        if event["request_id"] is not None
    ]

    checks_passed = (
        len(results) == len(records)
        and successful == len(records)
        and stored == len(records)
        and len(request_ids) == len(records)
        and len(set(request_ids)) == len(records)
    )

    latencies = pd.Series(
        [event["client_latency_ms"] for event in results]
    )

    summary = {
        "run_id": run_id,
        "scenario": args.scenario,
        "started_at": started_at,
        "finished_at": utc_now(),
        "api_url": base_url,
        "health_at_start": health,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "requested_count": len(records),
        "attempted_count": len(results),
        "http_200_count": successful,
        "stored_count": stored,
        "client_latency_median_ms": float(latencies.median()),
        "client_latency_p95_ms": float(latencies.quantile(0.95)),
        "checks_passed": checks_passed,
        "journal": str(journal_path.relative_to(PROJECT_ROOT)),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nHTTP 200 : {successful}/{len(records)}")
    print(f"Stockage confirmé par l'API : {stored}/{len(records)}")
    print(f"Latence médiane : {latencies.median():.2f} ms")
    print(f"Latence p95 : {latencies.quantile(0.95):.2f} ms")
    print(f"Vérifications réussies : {checks_passed}")
    print(f"Journal : {journal_path}")
    print(f"Résumé : {summary_path}")

    if not checks_passed:
        for event in results:
            if (
                event["status_code"] != 200
                or event["monitoring_status"] != "stored"
                or not event["request_id"]
            ):
                print("\nPremier appel problématique :")
                print(json.dumps(event, indent=2, ensure_ascii=False))
                break
        raise SystemExit(1)


if __name__ == "__main__":
    main()