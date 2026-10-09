import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    database_path = (
        tmp_path_factory.mktemp("monitoring")
        / "test_predictions.sqlite3"
    )

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(
            "MONITORING_DB_PATH",
            str(database_path),
        )

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def valid_payload():
    # Relire le fichier fournit une copie indépendante à chaque test.
    path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "valid_request.json"
    )

    with path.open(encoding="utf-8") as file:
        return json.load(file)