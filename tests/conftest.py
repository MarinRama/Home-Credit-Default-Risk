import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="session")
def client():
    # Le contexte déclenche le démarrage et le chargement du modèle.
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