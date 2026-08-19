"""Serve KIBATIN predictions from eagerly loaded specialized artifacts.

Importing this module loads the diagnosis, indication, and triage runtimes
before constructing the module-level Flask application.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request

from runtime import (
    PUBLIC_OPTIONS,
    RuntimeAssets,
    load_runtime,
    preprocess_input,
)


def get_top_predictions(
    probabilities: torch.Tensor,
    id2labels: dict[int, str],
    code_map: dict[str, str],
) -> list[dict[str, str | float]]:
    """Return the three highest-scoring labels in the public response schema.

    Each result contains the model label, its external code-map ID, and a
    probability rounded to four decimal places, ordered by descending score.
    """
    top_values, top_indices = torch.topk(probabilities, 3)
    response: list[dict[str, str | float]] = []
    for value, index in zip(top_values, top_indices):
        label = id2labels[int(index.item())]
        response.append(
            {
                "Name": label,
                "ID": code_map[label],
                "Prediction": round(float(value.item()), 4),
            }
        )
    return response


def create_app(runtime: RuntimeAssets) -> Flask:
    """Build the Flask application around a fully loaded runtime.

    Prediction and input failures are returned as HTTP 200 JSON error objects
    for compatibility; Flask retains its normal routing and method statuses.
    """
    app = Flask(__name__)

    @app.route("/healthz", methods=["GET"])
    def healthz():
        """Report readiness after all specialized runtimes have loaded."""
        return jsonify({"status": "ready"})

    @app.route("/predict", methods=["POST"])
    def predict():
        """Route a JSON request to the runtime selected by its public option.

        Successful responses contain the selected option and its top three
        predictions. Invalid options and caught preprocessing or inference
        failures produce an ``error`` object without changing the HTTP status.
        """
        try:
            data = request.get_json()
            option = data.get("option")
            if option not in PUBLIC_OPTIONS:
                return jsonify(
                    {
                        "error": (
                            f"Invalid option - {option} - valid options are "
                            f"{list(PUBLIC_OPTIONS)}"
                        )
                    }
                )

            with torch.inference_mode():
                mode_runtime = runtime.modes[option]
                inputs = preprocess_input(
                    data,
                    mode_runtime.tokenizer,
                    mode_runtime.preprocessing,
                    mode_runtime.device,
                )
                outputs = mode_runtime.model(**inputs)
                probabilities = F.softmax(
                    outputs[mode_runtime.internal_task_key], dim=1
                ).squeeze(0)
                response = get_top_predictions(
                    probabilities,
                    mode_runtime.id2labels,
                    mode_runtime.code_map,
                )
                return jsonify({option: response})
        except Exception as exc:
            return jsonify({"error": str(exc)})

    return app


GLOBAL_RUNTIME = load_runtime()
app = create_app(GLOBAL_RUNTIME)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
