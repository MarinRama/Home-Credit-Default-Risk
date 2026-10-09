import json
import sqlite3
from contextlib import closing

import pytest


def fetch_event(client, request_id):
    database_path = client.app.state.store.database_path

    with closing(sqlite3.connect(str(database_path))) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT *
            FROM prediction_events
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchall()

    assert len(rows) == 1
    return dict(rows[0])


def test_prediction_is_logged(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.headers["X-Monitoring-Status"] == "stored"

    event = fetch_event(
        client,
        response.headers["X-Request-ID"],
    )

    assert event["status_code"] == 200
    assert event["model_version"] == "1.0.0"
    assert json.loads(event["inputs_json"]) == valid_payload["features"]
    assert json.loads(event["outputs_json"]) == response.json()
    assert event["inference_ms"] >= 0
    assert event["processing_ms"] >= event["inference_ms"]
    assert event["error_type"] is None


@pytest.mark.parametrize(
    "payload, expected_error",
    [
        ({"features": {}}, "feature_validation_error"),
        ({}, "request_validation_error"),
    ],
)
def test_validation_errors_are_logged(client, payload, expected_error):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.headers["X-Monitoring-Status"] == "stored"

    event = fetch_event(
        client,
        response.headers["X-Request-ID"],
    )

    assert event["status_code"] == 422
    assert event["inference_ms"] is None
    assert event["outputs_json"] is None
    assert event["error_type"] == expected_error


def test_prediction_error_is_logged(client, valid_payload, monkeypatch):
    def failing_prediction(features, timings=None):
        raise RuntimeError("Erreur simulée")

    monkeypatch.setattr(
        client.app.state.predictor,
        "predict",
        failing_prediction,
    )

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500

    event = fetch_event(
        client,
        response.headers["X-Request-ID"],
    )

    assert event["status_code"] == 500
    assert event["error_type"] == "prediction_error"
    assert event["outputs_json"] is None


def test_storage_failure_is_reported(client, valid_payload, monkeypatch):
    def failing_record(**kwargs):
        raise sqlite3.OperationalError("Base indisponible")

    monkeypatch.setattr(
        client.app.state.store,
        "record",
        failing_record,
    )

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.headers["X-Monitoring-Status"] == "failed"
    assert response.headers["X-Request-ID"]