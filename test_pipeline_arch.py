"""
test_pipeline_arch.py
Architecture smoke tests — no trained weights required.

Run with:  pytest test_pipeline_arch.py -v
"""

import torch
import pytest
from pipeline import build_pipeline
from config import BERT_DIM, BOTTLENECK_DIM, NUM_CLASSES

BATCH = 8
DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def pipeline():
    p = build_pipeline(use_noise=True, device=DEVICE)
    p.train()   # BatchNorm needs train mode for a batch > 1
    return p


def test_classifier_reads_768d_not_64d(pipeline):
    """The classifier's first linear layer must accept 768-D input, not the 64-D bottleneck."""
    assert pipeline.decoder.input_proj.in_features == BERT_DIM, (
        f"Expected {BERT_DIM}, got {pipeline.decoder.input_proj.in_features} — "
        "did you forget TaskDecoder(input_dim=BERT_DIM) in build_pipeline()?"
    )


def test_forward_train_shapes(pipeline):
    """forward_train must return (logits [B, NUM_CLASSES], x_hat [B, 768])."""
    x = torch.randn(BATCH, BERT_DIM)
    logits, x_hat = pipeline.forward_train(x)
    assert logits.shape == (BATCH, NUM_CLASSES)
    assert x_hat.shape == (BATCH, BERT_DIM)


def test_forward_inference_shape(pipeline):
    """Inference forward must return logits [B, NUM_CLASSES]."""
    pipeline.eval()
    x = torch.randn(BATCH, BERT_DIM)
    with torch.no_grad():
        logits = pipeline(x)
    assert logits.shape == (BATCH, NUM_CLASSES)
    pipeline.train()


def test_reconstruction_is_768d_not_bottleneck(pipeline):
    """x_hat returned by forward_train must be 768-D (full embedding), not 64-D bottleneck."""
    x = torch.randn(BATCH, BERT_DIM)
    _, x_hat = pipeline.forward_train(x)
    assert x_hat.shape[-1] == BERT_DIM, (
        f"Reconstruction is {x_hat.shape[-1]}-D — expected {BERT_DIM}-D"
    )
    assert x_hat.shape[-1] != BOTTLENECK_DIM


def test_reconstruction_depends_on_channel_noise():
    """x_hat must differ between noisy and noiseless pipelines (decoder is in the signal path)."""
    p_noisy = build_pipeline(use_noise=True, device=DEVICE)
    p_clean = build_pipeline(use_noise=False, device=DEVICE)

    # copy weights so the only difference is the channel
    p_clean.autoencoder.load_state_dict(p_noisy.autoencoder.state_dict())
    p_clean.decoder.load_state_dict(p_noisy.decoder.state_dict())

    p_noisy.train()
    p_clean.train()

    x = torch.randn(BATCH, BERT_DIM)
    with torch.no_grad():
        _, x_hat_noisy = p_noisy.forward_train(x)
        _, x_hat_clean = p_clean.forward_train(x)

    assert not torch.allclose(x_hat_noisy, x_hat_clean), (
        "Noisy and clean reconstructions are identical — "
        "the AE decoder may not be in the channel signal path."
    )
