import pytest
from app import app  # importe l'objet Flask défini dans app.py

@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8").lower()
    assert "prédiction de défaut" in html or "pr\u00e9diction de d\u00e9faut" in html

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
    assert resp.status_code == 200
    html = resp.data.decode("utf-8").lower()
    assert "probabilité de défaut" in html or "probabilit\u00e9 de d\u00e9faut" in html
    assert "prédiction" in html or "pr\u00e9diction" in html
