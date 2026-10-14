import json
import sqlite3
from contextlib import closing

import pytest

from src.monitoring.storage import PredictionStore


def make_event():
    return {
        "request_id": "test-request-001",
        "created_at": "2026-10-09T10:00:00+00:00",
        "model_version": "1.0.0",
        "status_code": 200,
        "inputs": {
            "AMT_CREDIT": 100000.0,
            "EXT_SOURCE_1": None,
        },
        "outputs": {
            "score": 0.08,
            "prediction": 0,
            "threshold": 0.5,
        },
        "inference_ms": 2.0,
        "processing_ms": 5.0,
        "error_type": None,
    }


def read_event(database_path):
    with closing(sqlite3.connect(str(database_path))) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            "SELECT * FROM prediction_events"
        ).fetchone()

        return dict(row)


def test_successful_event_is_persisted(tmp_path):
    database_path = tmp_path / "production.sqlite3"
    store = PredictionStore(database_path)

    event = make_event()
    store.record(**event)

    # Recréer l'objet simule la réouverture du stockage.
    PredictionStore(database_path)

    saved = read_event(database_path)

    assert saved["request_id"] == event["request_id"]
    assert saved["created_at"] == event["created_at"]
    assert saved["model_version"] == "1.0.0"
    assert saved["status_code"] == 200
    assert json.loads(saved["inputs_json"]) == event["inputs"]
    assert json.loads(saved["outputs_json"]) == event["outputs"]
    assert saved["inference_ms"] == 2.0
    assert saved["processing_ms"] == 5.0
    assert saved["error_type"] is None


def test_error_event_is_persisted(tmp_path):
    database_path = tmp_path / "production.sqlite3"
    store = PredictionStore(database_path)

    event = make_event()
    event.update(
        status_code=422,
        inputs={"AMT_CREDIT": "incorrect"},
        outputs=None,
        inference_ms=None,
        error_type="validation_error",
    )

    store.record(**event)
    saved = read_event(database_path)

    assert saved["status_code"] == 422
    assert saved["outputs_json"] is None
    assert saved["inference_ms"] is None
    assert saved["error_type"] == "validation_error"
    assert json.loads(saved["inputs_json"]) == event["inputs"]


def test_duplicate_request_id_is_rejected(tmp_path):
    database_path = tmp_path / "production.sqlite3"
    store = PredictionStore(database_path)

    event = make_event()
    store.record(**event)

    with pytest.raises(sqlite3.IntegrityError):
        store.record(**event)

    with closing(sqlite3.connect(str(database_path))) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM prediction_events"
        ).fetchone()[0]

    assert count == 1