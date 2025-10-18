from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from logging.config import dictConfig

dictConfig({
    "version": 1,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        }
    },
    "handlers": {
        "wsgi": {
            "class": "logging.StreamHandler",
            "stream": "ext://flask.logging.wsgi_errors_stream",
            "formatter": "default",
        }
    },
    "root": {
        "level": "INFO",  
        "handlers": ["wsgi"]
    }
})


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, "notebook", "models", "best_model.pkl")
bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
scaler = bundle.get("scaler", None)
feature_order = bundle["feature_order"]

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    # attend {"features": {"feat1": val, ...}} ou {"instances": [{...}, {...}]}
    if "features" in payload:
        instances = [payload["features"]]
    else:
        instances = payload.get("instances", [])
    if not instances:
        return jsonify({"error": "No features provided"}), 400

    # Construction X dans le bon ordre de colonnes
    try:
        X = np.array([[inst[col] for col in feature_order] for inst in instances], dtype=float)
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400

    # Standardisation
    if scaler is not None:
        X = scaler.transform(X)

    proba = model.predict_proba(X)[:, 1].tolist()
    pred = model.predict(X).tolist()
    return jsonify({"proba_default": proba, "pred": pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=True)

print("CWD:", os.getcwd())
print("Looking for:", BUNDLE_PATH)
print("Models dir exists?", os.path.isdir(os.path.dirname(BUNDLE_PATH)))

