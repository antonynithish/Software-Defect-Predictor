
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file
)

from app.predictor import (
    predict_defect,
    FEATURES,
    THRESHOLD
)

from app.database import (
    init_db,
    save_prediction,
    get_prediction_history
)

from app.explainer import explain_prediction

from app.analytics import (
    get_analytics,
    get_figure_path
)


app = Flask(__name__)

init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "application": "Automated Software Bug & Defect Predictor",
        "model": "KC1 Grouped Random Forest"
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model": "KC1 Grouped Random Forest",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "classification_threshold": THRESHOLD
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide a valid JSON object."
        }), 400

    metrics = data.get("metrics", data)

    if not isinstance(metrics, dict):
        return jsonify({
            "error": "Metrics must be provided as an object."
        }), 400

    try:
        result = predict_defect(metrics)

        prediction_id = save_prediction(
            metrics,
            result
        )

        result["prediction_id"] = prediction_id

        return jsonify(result), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Check the server logs."
        }), 500


@app.route("/api/explain", methods=["POST"])
def explain():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide a valid JSON object."
        }), 400

    metrics = data.get("metrics", data)

    if not isinstance(metrics, dict):
        return jsonify({
            "error": "Metrics must be provided as an object."
        }), 400

    try:
        result = explain_prediction(metrics)

        return jsonify(result), 200

    except ValueError as error:
        app.logger.exception("SHAP explanation failed")

        return jsonify({
            "error": str(error)
        }), 422

    except Exception:
        app.logger.exception("SHAP explanation failed")

        return jsonify({
            "error": "Unable to generate an individual SHAP explanation."
        }), 500


@app.route("/api/history", methods=["GET"])
def history():
    try:
        limit = request.args.get(
            "limit",
            default=50,
            type=int
        )

        if limit is None or limit < 1 or limit > 200:
            return jsonify({
                "error": "Limit must be between 1 and 200."
            }), 400

        return jsonify({
            "history": get_prediction_history(limit)
        })

    except Exception:
        app.logger.exception(
            "Could not retrieve prediction history"
        )

        return jsonify({
            "error": "Could not retrieve prediction history."
        }), 500


@app.route("/api/analytics", methods=["GET"])
def analytics():
    try:
        return jsonify(get_analytics()), 200

    except Exception:
        app.logger.exception(
            "Could not retrieve model analytics"
        )

        return jsonify({
            "error": "Could not retrieve model analytics."
        }), 500


@app.route(
    "/api/analytics/figure/<filename>",
    methods=["GET"]
)
def analytics_figure(filename):
    path = get_figure_path(filename)

    if path is None:
        return jsonify({
            "error": "Requested evaluation figure was not found."
        }), 404

    return send_file(path, mimetype="image/png")


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
