import pytest


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert body["n_features"] == 541
    assert len(body["feature_names"]) == 541
    assert len(set(body["feature_names"])) == 541


def test_valid_prediction(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    body = response.json()

    # Score déjà vérifié sur le client 208550.
    assert body["score"] == pytest.approx(
        0.0884589283752367,
        rel=0,
        abs=1e-10,
    )
    assert body["prediction"] == 0
    assert body["threshold"] == pytest.approx(0.5)
    assert body["model_version"] == "1.0.0"


def test_feature_order_does_not_change_score(client, valid_payload):
    original = client.post("/predict", json=valid_payload)

    valid_payload["features"] = dict(
        reversed(list(valid_payload["features"].items()))
    )

    reordered = client.post("/predict", json=valid_payload)

    assert original.status_code == 200
    assert reordered.status_code == 200
    assert reordered.json()["score"] == original.json()["score"]


def test_missing_feature(client, valid_payload):
    valid_payload["features"].pop("AMT_CREDIT")

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert "AMT_CREDIT" in response.json()["detail"]


def test_unexpected_feature(client, valid_payload):
    valid_payload["features"]["UNKNOWN_FEATURE"] = 123

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert "UNKNOWN_FEATURE" in response.json()["detail"]


@pytest.mark.parametrize("invalid_value", ["texte", "123", True, [], {}])
def test_invalid_feature_type(client, valid_payload, invalid_value):
    valid_payload["features"]["AMT_CREDIT"] = invalid_value

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert "AMT_CREDIT" in response.json()["detail"]


def test_null_value_is_accepted(client, valid_payload):
    valid_payload["features"]["EXT_SOURCE_1"] = None

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert 0 <= response.json()["score"] <= 1


def test_all_null_values_are_rejected(client, valid_payload):
    valid_payload["features"] = {
        name: None for name in valid_payload["features"]
    }

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [{}, {"features": {}}, {"features": []}, {"features": None}],
)
def test_invalid_request_structure(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_unexpected_top_level_field(client, valid_payload):
    valid_payload["unexpected"] = "value"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_internal_error_is_controlled(client, valid_payload, monkeypatch):
    predictor = client.app.state.predictor

    def failing_prediction(features, timings=None):
        raise RuntimeError("Détail technique interne")

    monkeypatch.setattr(predictor, "predict", failing_prediction)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Une erreur interne empêche le calcul du score."
    )
    assert "Détail technique interne" not in response.text


def test_unavailable_model(client, monkeypatch):
    monkeypatch.setattr(client.app.state, "predictor", None)

    response = client.get("/health")

    assert response.status_code == 503

@pytest.mark.parametrize(
    "feature, invalid_value",
    [
        ("AMT_CREDIT", -100),
        ("AMT_INCOME_TOTAL", -1),
        ("AMT_ANNUITY", -10),
        ("AMT_GOODS_PRICE", -1),
        ("CNT_CHILDREN", -1),
        ("CNT_CHILDREN", 1.5),
        ("CNT_FAM_MEMBERS", 0),
        ("CNT_FAM_MEMBERS", 2.5),
        ("OWN_CAR_AGE", -1),
        ("AGE_YEARS", -5),
        ("EXT_SOURCE_1", -0.1),
        ("EXT_SOURCE_2", 1.1),
        ("EXT_SOURCE_3", 2),
    ],
)
def test_out_of_range_values(
    client,
    valid_payload,
    feature,
    invalid_value,
):
    valid_payload["features"][feature] = invalid_value

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert feature in response.json()["detail"]


@pytest.mark.parametrize("value", [0, 1, None])
def test_external_score_boundaries(client, valid_payload, value):
    valid_payload["features"]["EXT_SOURCE_1"] = value

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert 0 <= response.json()["score"] <= 1