import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.scoring.predictor import CreditPredictor


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement unique au démarrage de chaque processus serveur.
    app.state.predictor = CreditPredictor()
    yield
    app.state.predictor = None


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
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas disponible.",
        )

    return predictor


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
    predictor = get_predictor(request)

    # Séparer les erreurs de saisie des erreurs internes du modèle.
    try:
        predictor.prepare_input(payload.features)
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    try:
        return predictor.predict(payload.features)
    except Exception as error:
        logger.exception("Échec du calcul de la prédiction")

        raise HTTPException(
            status_code=500,
            detail="Une erreur interne empêche le calcul du score.",
        ) from error