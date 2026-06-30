"""
predict.py
Classify a sentence through the full pipeline:
    BERT -> AE encoder -> quantizer -> channel -> decoder -> class

Usage:
    python predict.py                          # interactive, no noise
    python predict.py --noise                  # interactive, with AWGN
    python predict.py "Your sentence here"     # single sentence, no noise
    python predict.py --noise "Your sentence"  # single sentence, with AWGN
"""

import sys
import torch

from bert_encoder import BERTEncoder
from pipeline import build_pipeline
from config import MODEL_FILE, CATEGORY_NAMES, NOISE_STD, snr_db_from_std


def load_system(use_noise: bool):
    print("[INFO] Loading BERT encoder...")
    bert = BERTEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Loading trained pipeline...")
    pipeline = build_pipeline(use_noise=use_noise, device=device, checkpoint_path=MODEL_FILE)
    pipeline.eval()
    return bert, pipeline, device


def predict(sentence, bert, pipeline, device):
    embedding = bert.encode_texts([sentence]).to(device)
    with torch.no_grad():
        probs = torch.softmax(pipeline(embedding), dim=1)[0]
    return probs.argmax().item(), probs


def print_result(sentence, pred, probs, use_noise):
    label = f"AWGN (std={NOISE_STD}, ~{snr_db_from_std(NOISE_STD):.1f} dB)" if use_noise else "Clean"
    print(f"\n{'-'*55}")
    print(f"  Sentence : {sentence}")
    print(f"  Channel  : {label}")
    print(f"  Result   : {CATEGORY_NAMES[pred]}")
    print(f"\n  Confidence:")
    for i, name in enumerate(CATEGORY_NAMES):
        bar = "#" * int(probs[i].item() * 30)
        marker = "  <- predicted" if i == pred else ""
        print(f"    {name:<10} {probs[i].item()*100:5.1f}%  {bar}{marker}")
    print(f"{'-'*55}\n")


def main():
    args = sys.argv[1:]
    use_noise = "--noise" in args
    if use_noise:
        args.remove("--noise")

    bert, pipeline, device = load_system(use_noise)

    if args:
        sentence = " ".join(args)
        pred, probs = predict(sentence, bert, pipeline, device)
        print_result(sentence, pred, probs, use_noise)
        return

    mode = "WITH noise" if use_noise else "WITHOUT noise"
    print(f"\n[INFO] Interactive mode ({mode}). Type 'quit' to exit.\n")
    while True:
        sentence = input("Sentence: ").strip()
        if sentence.lower() in ("quit", "exit", "q"):
            break
        if not sentence:
            continue
        pred, probs = predict(sentence, bert, pipeline, device)
        print_result(sentence, pred, probs, use_noise)


if __name__ == "__main__":
    main()
