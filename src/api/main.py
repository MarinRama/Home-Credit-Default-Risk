import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from src.monitoring.storage import PredictionStore
from src.scoring.predictor import CreditPredictor, InputValidationError


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = CreditPredictor()

    database_path = os.getenv(
        "MONITORING_DB_PATH",
        str(
            PROJECT_ROOT / "data" / "production"
            / "predictions.sqlite3"
        ),
    )

    app.state.store = PredictionStore(database_path)

    yield

    app.state.predictor = None
    app.state.store = None


app = FastAPI(
    title="API de scoring — Prêt à Dépenser",
    description=(
        "Calcule un score de risque de défaut à partir des "
        "541 variables préparées d'un client."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: Dict[str, Any] = Field(
        ...,
        description=(
            "Les 541 variables avec les noms du modèle. "
            "Utiliser null pour une valeur manquante."
        ),
    )


class PredictionResponse(BaseModel):
    score: float = Field(..., ge=0, le=1)
    prediction: int = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0, le=1)
    model_version: str


def get_predictor(request: Request) -> CreditPredictor:
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        request.state.error_type = "model_unavailable"

        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas disponible.",
        )

    return predictor


def serializable_inputs(value):
    """Évite qu'un corps invalide empêche la journalisation."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return None

    return value


@app.middleware("http")
async def monitor_predictions(request: Request, call_next):
    # Seuls les appels de scoring alimentent cette table.
    if request.method != "POST" or request.url.path != "/predict":
        return await call_next(request)

    start = perf_counter()
    request_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    request.state.inputs = None
    request.state.outputs = None
    request.state.timings = {}
    request.state.error_type = None

    predictor = getattr(request.app.state, "predictor", None)
    model_version = (
        predictor.metadata["model_version"]
        if predictor is not None
        else None
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Erreur inattendue pendant la requête")
        request.state.error_type = "internal_error"

        response = JSONResponse(
            status_code=500,
            content={"detail": "Une erreur interne est survenue."},
        )

    # Mesure avant écriture SQLite, hors transport réseau.
    processing_ms = (perf_counter() - start) * 1000

    event = {
        "request_id": request_id,
        "created_at": created_at,
        "model_version": model_version,
        "status_code": response.status_code,
        "inputs": serializable_inputs(request.state.inputs),
        "outputs": request.state.outputs,
        "inference_ms": request.state.timings.get("inference_ms"),
        "processing_ms": processing_ms,
        "error_type": request.state.error_type,
    }

    try:
        # SQLite est synchrone : exécution hors de la boucle async.
        await run_in_threadpool(
            request.app.state.store.record,
            **event,
        )
        monitoring_status = "stored"
    except Exception:
        logger.exception(
            "Échec du stockage de la requête %s",
            request_id,
        )
        monitoring_status = "failed"

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Monitoring-Status"] = monitoring_status

    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    error: RequestValidationError,
):
    body = error.body

    request.state.inputs = (
        body.get("features")
        if isinstance(body, dict) and "features" in body
        else body
    )
    request.state.error_type = "request_validation_error"

    # Réponse contrôlée, sans renvoyer le corps brut invalide.
    details = [
        {
            "loc": list(item["loc"]),
            "msg": item["msg"],
            "type": item["type"],
        }
        for item in error.errors()
    ]

    return JSONResponse(
        status_code=422,
        content={"detail": details},
    )


@app.get("/health", summary="Vérifier la disponibilité du service")
def health(request: Request):
    predictor = get_predictor(request)

    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": predictor.metadata["model_version"],
    }


@app.get("/model-info", summary="Consulter le contrat du modèle")
def model_info(request: Request):
    predictor = get_predictor(request)

    return {
        "model_version": predictor.metadata["model_version"],
        "model_type": predictor.metadata["model_type"],
        "n_features": len(predictor.feature_names),
        "threshold": predictor.threshold,
        "feature_names": predictor.feature_names,
        "classes": {
            "0": "Absence de défaut prédite",
            "1": "Défaut prédit",
        },
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Calculer le score d'un client",
)
def predict(payload: PredictionRequest, request: Request):
    request.state.inputs = payload.features
    predictor = get_predictor(request)

    try:
        result = predictor.predict(
            payload.features,
            timings=request.state.timings,
        )
        request.state.outputs = result
        return result

    except InputValidationError as error:
        request.state.error_type = "feature_validation_error"

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        request.state.error_type = "prediction_error"
        logger.exception("Échec du calcul de la prédiction")

        raise HTTPException(
            status_code=500,
            detail="Une erreur interne empêche le calcul du score.",
        ) from error