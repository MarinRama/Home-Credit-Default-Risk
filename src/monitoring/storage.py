import json
import sqlite3
from contextlib import closing
from pathlib import Path


class PredictionStore:
    def __init__(self, database_path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialize()

    def connect(self):
        # Une connexion par opération pour éviter de partager
        # une connexion entre les threads de l'API.
        return sqlite3.connect(
            str(self.database_path),
            timeout=10,
        )

    def initialize(self):
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_events (
                        request_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        model_version TEXT,
                        status_code INTEGER NOT NULL,
                        inputs_json TEXT,
                        outputs_json TEXT,
                        inference_ms REAL,
                        processing_ms REAL NOT NULL,
                        error_type TEXT
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_prediction_events_created_at
                    ON prediction_events (created_at)
                    """
                )

    def record(
        self,
        *,
        request_id,
        created_at,
        model_version,
        status_code,
        inputs,
        outputs,
        inference_ms,
        processing_ms,
        error_type=None,
    ):
        inputs_json = (
            json.dumps(
                inputs,
                ensure_ascii=False,
                allow_nan=False,
            )
            if inputs is not None
            else None
        )

        outputs_json = (
            json.dumps(
                outputs,
                ensure_ascii=False,
                allow_nan=False,
            )
            if outputs is not None
            else None
        )

        values = (
            request_id,
            created_at,
            model_version,
            status_code,
            inputs_json,
            outputs_json,
            inference_ms,
            processing_ms,
            error_type,
        )

        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO prediction_events (
                        request_id,
                        created_at,
                        model_version,
                        status_code,
                        inputs_json,
                        outputs_json,
                        inference_ms,
                        processing_ms,
                        error_type
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )