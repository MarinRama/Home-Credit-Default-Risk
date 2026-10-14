import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8001",
).rstrip("/")


def call_api(path, payload=None):
    data = (
        json.dumps(payload, allow_nan=False).encode("utf-8")
        if payload is not None
        else None
    )

    request = Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )

    try:
        response = urlopen(request, timeout=30)
    except HTTPError as error:
        response = error

    with response:
        return (
            response.status,
            json.loads(response.read().decode("utf-8")),
            dict(response.headers.items()),
        )


def main():
    status, health, _ = call_api("/health")

    assert status == 200, (status, health)
    assert health["model_loaded"] is True

    fixture_path = (
        PROJECT_ROOT / "tests" / "fixtures"
        / "valid_request.json"
    )

    with fixture_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    status, prediction, headers = call_api("/predict", payload)
    headers = {key.lower(): value for key, value in headers.items()}

    assert status == 200, (status, prediction)
    assert math.isclose(
        prediction["score"],
        0.0884589283752367,
        rel_tol=0,
        abs_tol=1e-10,
    ), prediction
    assert prediction["prediction"] == 0
    assert headers.get("x-monitoring-status") == "stored"
    assert headers.get("x-request-id")

    error_status, error_body, error_headers = call_api(
        "/predict",
        {"features": {}},
    )
    error_headers = {
        key.lower(): value
        for key, value in error_headers.items()
    }

    assert error_status == 422, (error_status, error_body)
    assert error_headers.get("x-monitoring-status") == "stored"
    assert error_headers.get("x-request-id")
    assert (
        error_headers["x-request-id"]
        != headers["x-request-id"]
    )

    report = {
        "health": health,
        "prediction": prediction,
        "valid_request_id": headers["x-request-id"],
        "invalid_request_id": error_headers["x-request-id"],
        "invalid_request_status": error_status,
        "checks_passed": True,
    }

    output_dir = PROJECT_ROOT / "reports" / "deployment"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "smoke_test.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Vérification du déploiement : OK")


if __name__ == "__main__":
    main()