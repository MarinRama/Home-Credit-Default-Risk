import numpy as np
import pytest


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_values_are_rejected(client, valid_payload, value):
    predictor = client.app.state.predictor
    valid_payload["features"]["AMT_CREDIT"] = value

    with pytest.raises(ValueError, match="AMT_CREDIT"):
        predictor.prepare_input(valid_payload["features"])


@pytest.mark.parametrize(
    "offset, expected_prediction",
    [
        (-0.000001, 0),
        (0.0, 1),
        (0.000001, 1),
    ],
)
def test_decision_threshold(
    client,
    valid_payload,
    monkeypatch,
    offset,
    expected_prediction,
):
    predictor = client.app.state.predictor
    controlled_score = predictor.threshold + offset

    def fake_predict_proba(data):
        return np.array([[1 - controlled_score, controlled_score]])

    monkeypatch.setattr(
        predictor.model,
        "predict_proba",
        fake_predict_proba,
    )

    result = predictor.predict(valid_payload["features"])

    assert result["score"] == controlled_score
    assert result["prediction"] == expected_prediction