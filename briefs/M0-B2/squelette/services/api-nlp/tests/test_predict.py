"""Tests pour l'endpoint POST /predict.

Ces tests appellent le coeur d'inférence réel (pipeline HF chargé
via le lifespan de l'app). Ils peuvent être plus lents.
"""
from __future__ import annotations

import csv
from pathlib import Path
from urllib import response

from fastapi.testclient import TestClient
import pytest

from app.main import app, MAX_TEXT_LENGTH, MODEL_NAME

def test_predict_endpoint_cas_valide_returns_200() -> None:
    """Cas valide : appel réel au pipeline, on attend un sentiment négatif."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"texte": "Chambre sale, accueil terrible."})

    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "négatif"
    assert "scores_5_stars" in body
    assert body["model_name"] == MODEL_NAME


def test_predict_endpoint_text_vide_returns_422() -> None:
	"""Texte vide ou uniquement espaces -> pydantic renvoie 422."""
	with TestClient(app) as client:
		response = client.post("/predict", json={"texte": ""})

	assert response.status_code == 422


def test_predict_endpoint_text_trop_long_returns_422() -> None:
	"""Texte au-delà de MAX_TEXT_LENGTH -> 422 levée par la route."""
	long_text = "a" * (MAX_TEXT_LENGTH + 1)
	with TestClient(app) as client:
		response = client.post("/predict", json={"texte": long_text})

	assert response.status_code == 422


@pytest.mark.parametrize("row_id", [2, 3, 1])
def test_predict_endpoint_param_with_data(row_id: int) -> None:
    """Test paramétré : on lit `data/sample_reviews.csv` et on vérifie
    que l'API renvoie une structure valide pour trois exemples (pos/neu/neg).
    Appel réel au pipeline.
    """
    csv_path = Path(__file__).parent / "data" / "sample_reviews.csv"
    assert csv_path.exists(), f"Fichier de test introuvable: {csv_path}"

    expected = None
    texte = None
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row["id"]) == row_id:
                texte = row["texte"]
                expected = row["sentiment_attendu"]
                break

    assert texte is not None and expected is not None

    with TestClient(app) as client:
        response = client.post("/predict", json={"texte": texte})

    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == expected