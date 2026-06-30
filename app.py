"""
app.py
Flask web interface for the semantic communication pipeline.

    python app.py   ->   http://localhost:5000

Loads two pipelines once (clean + noisy) and routes classification requests.
"""

import torch
from flask import Flask, render_template, request, jsonify

from bert_encoder import BERTEncoder
from pipeline import build_pipeline
from config import (
    MODEL_FILE,
    CATEGORY_NAMES,
    CATEGORY_ICONS,
    NOISE_STD,
    QAM_ORDER,
    snr_db_from_std,
)

app = Flask(__name__)

print("[INFO] Loading BERT encoder...")
bert = BERTEncoder()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

print("[INFO] Loading clean pipeline...")
pipeline_clean = build_pipeline(use_noise=False, device=device, checkpoint_path=MODEL_FILE)
pipeline_clean.eval()

print("[INFO] Loading noisy pipeline...")
pipeline_noisy = build_pipeline(use_noise=True, device=device, checkpoint_path=MODEL_FILE)
pipeline_noisy.eval()

print("[INFO] Ready.\n")


@app.route("/")
def index():
    return render_template(
        "index.html",
        noise_std=NOISE_STD,
        snr_db=round(snr_db_from_std(NOISE_STD), 1),
        qam_order=QAM_ORDER,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sentence = data.get("sentence", "").strip()
    use_noise = data.get("noise", False)

    if not sentence:
        return jsonify({"error": "Please enter a sentence."}), 400

    embedding = bert.encode_texts([sentence]).to(device)
    pipeline = pipeline_noisy if use_noise else pipeline_clean

    with torch.no_grad():
        probs = torch.softmax(pipeline(embedding), dim=1)[0].tolist()

    pred = probs.index(max(probs))
    return jsonify(
        {
            "prediction": CATEGORY_NAMES[pred],
            "icon": CATEGORY_ICONS[pred],
            "scores": [
                {
                    "label": CATEGORY_NAMES[i],
                    "icon": CATEGORY_ICONS[i],
                    "prob": round(probs[i] * 100, 1),
                }
                for i in range(len(CATEGORY_NAMES))
            ],
            "noise": use_noise,
            "noise_std": NOISE_STD if use_noise else None,
        }
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
