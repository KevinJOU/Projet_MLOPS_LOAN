# 1) Imports
from flask import Flask, request, render_template
import os, joblib, numpy as np
from logging.config import dictConfig

# 2) Logging + app
dictConfig({
    "version": 1,
    "formatters": {"default": {"format": "[%(asctime)s] %(levelname)s: %(message)s"}},
    "handlers": {"wsgi": {"class": "logging.StreamHandler", "formatter": "default"}},
    "root": {"level": "INFO", "handlers": ["wsgi"]},
})
app = Flask(__name__)

# 3) Chargement du bundle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, "notebook", "models", "best_model.pkl")  # ajuste si besoin
bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
scaler = bundle.get("scaler", None)
feature_order = bundle["feature_order"]

# Endpoints déjà existants (exemples)
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/version", methods=["GET"])
def version():
    return {"model_version": "v0.1.0"}, 200

# 4) AJOUTE ICI tes deux routes HTML
@app.route("/", methods=["GET"])
def index_page():
    return render_template("index.html")

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        feats = {
            "credit_lines_outstanding": float(request.form["credit_lines_outstanding"]),
            "loan_amt_outstanding": float(request.form["loan_amt_outstanding"]),
            "total_debt_outstanding": float(request.form["total_debt_outstanding"]),
            "income": float(request.form["income"]),
            "years_employed": float(request.form["years_employed"]),
            "fico_score": float(request.form["fico_score"]),
        }
    except Exception as e:
        app.logger.warning("Form parsing error: %s", e)
        return render_template("index.html", result={"error": "Saisie invalide"})
    
    except KeyError as e:
        app.logger.warning("Champ manquant: %s", e)
        return render_template("index.html", result={"error": "Champ manquant"})
    except ValueError as e:
        app.logger.warning("Valeur non numérique: %s", e)
        return render_template("index.html", result={"error": "Valeur non numérique"})

    X = np.array([[feats[col] for col in feature_order]], dtype=float)
    if scaler is not None:
        X = scaler.transform(X)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(model.predict(X)[0])
    return render_template("index.html", result={"proba": round(proba, 4), "pred": pred})

# 5) Démarrage
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=True)