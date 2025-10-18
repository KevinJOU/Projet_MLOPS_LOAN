import os
import json
import pytest

# IMPORTANT: définir la variable d'env pour que Flask sache où est l'app si besoin
# Ici, on importe directement app depuis app.py
from app import app  # noqa: E402

@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status" in data

def test_index_page(client):
    resp = client.get("/")
    # Doit renvoyer du HTML (200 OK)
    assert resp.status_code == 200
    assert b"Pr\u00e9diction de d\u00e9faut" in resp.data or b"Pr\xc3\xa9diction de d\xc3\xa9faut" in resp.data

def test_predict_form_ok(client):
    form = {
        "credit_lines_outstanding": "2",
        "loan_amt_outstanding": "8000",
        "total_debt_outstanding": "12000",
        "income": "36000",
        "years_employed": "5",
        "fico_score": "670",
    }
    resp = client.post("/predict_form", data=form, follow_redirects=True)
    # La page HTML doit contenir "Résultat" si tout va bien
    assert resp.status_code == 200
    assert b"R\u00e9sultat" in resp.data or b"R\xc3\xa9sultat" in resp.data

def test_predict_form_missing_feature(client):
    form = {
        # On enlève une feature pour vérifier la gestion d'erreur
        "loan_amt_outstanding": "8000",
        "total_debt_outstanding": "12000",
        "income": "36000",
        "years_employed": "5",
        "fico_score": "670",
    }
    resp = client.post("/predict_form", data=form, follow_redirects=True)
    # Selon ton implémentation, tu renvoies une page avec un message d'erreur
    assert resp.status_code == 200
    # Cherche le message "Saisie invalide" ou autre feedback
    assert b"Saisie invalide" in resp.data or b"Erreur" in resp.data
