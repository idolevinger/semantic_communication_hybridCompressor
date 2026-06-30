"""
config.py
Global configuration for the semantic communication project.

Single source of truth: every other module imports from here.
Do NOT redefine any of these constants elsewhere.

Pipeline:
    Text -> BERT -> Autoencoder -> Quantizer -> AWGN Channel -> Decoder -> Classification
"""

import os

# -------------------- Project paths --------------------
# NOTE: config.py lives flat in the project root, so PROJECT_ROOT is a single
# dirname() of this file. (If you ever move config.py into a src/ subfolder,
# change this to dirname(dirname(...)).)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cached files
CACHE_TEXTS_FILE = os.path.join(DATA_DIR, "stage1_texts.csv")        # raw text + labels
CACHE_EMBEDDINGS_FILE = os.path.join(DATA_DIR, "stage1_embeddings.pt")  # 768-D BERT vectors
MODEL_FILE = os.path.join(RESULTS_DIR, "trained_pipeline.pt")

# -------------------- Data configuration --------------------
# Use the fully qualified Hugging Face dataset repo name. Some newer versions of
# datasets/huggingface_hub reject the short alias "ag_news".
DATASET_NAME = "fancyzhx/ag_news"
TRAIN_SIZE = 50000          # samples drawn from AG News train split
TEST_SIZE = 7600            # full AG News test split
VAL_RATIO = 0.2             # fraction of train used for validation
MAX_SEQ_LEN = 128
RANDOM_SEED = 42
USE_CACHED = True           # load cached text/embeddings if present

# Class names (AG News label order) and display icons
CATEGORY_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
CATEGORY_ICONS = ["\U0001F30D", "\u26BD", "\U0001F4BC", "\U0001F52C"]

# -------------------- Architecture dimensions --------------------
BERT_MODEL_NAME = "bert-base-uncased"
BERT_DIM = 768              # CLS embedding dimension out of BERT
BOTTLENECK_DIM = 64         # autoencoder output dim (== decoder input dim)
HIDDEN_DIM = 128            # decoder hidden size
NUM_CLASSES = 4

# -------------------- Quantizer / constellation --------------------
# Square-QAM constellation. QAM_ORDER (M) must be a perfect square (4, 16, 64, 256...).
# Each pair of real bottleneck dims is treated as one complex (I/Q) symbol, so
# BOTTLENECK_DIM must be even and the number of transmitted symbols is BOTTLENECK_DIM // 2.
QAM_ORDER = 16              # M: try 16, 64, 256 -- higher M = lower quantization error
QAM_POWER = 1.0            # average constellation symbol power (P-bar)
LEARNED_CONSTELLATION = False  # False = fixed square QAM; True = trainable points

# Soft-to-hard "hardness" parameter (sigma_q) annealing schedule.
# Annealed per OPTIMIZER STEP (not per epoch): sigma_q = min(MAX, INIT + RATE * step).
# sigma_q must grow LARGE for the soft (trained) assignment to match the hard
# (transmitted) assignment -- if it stays small, there is a train/transmit mismatch.
# Tune ANNEAL_RATE to your training budget: with ~390 steps/epoch (50k samples,
# batch 128), the values below reach sigma_q ~= 100 well within 50 epochs. If you
# train far longer or shorter, scale this accordingly.
SOFT_Q_INIT = 5.0
SOFT_Q_MAX = 100.0
SOFT_Q_ANNEAL_RATE = 5e-3            # added per step (reaches MAX in ~19k steps)

# KL regularizer weight (encourages uniform use of constellation points).
# Paper uses 0.05 for small M, 0 for very large M (>= 4096).
KL_LAMBDA = 0.05

# -------------------- Channel noise --------------------
NOISE_APPLY_TRAIN = True    # inject AWGN during training
NOISE_APPLY_EVAL = True     # inject AWGN during evaluation
NOISE_MEAN = 0.0
# IMPORTANT: because the autoencoder output is power-normalized and then quantized
# to a unit-power constellation, NOISE_STD now corresponds to a well-defined SNR.
# These values are NOT comparable to an older unnormalized-bottleneck setup.
#   SNR(dB) = 10 * log10(QAM_POWER / NOISE_STD**2)
# e.g. with QAM_POWER=1.0: std=0.1 -> 20 dB, std=0.3 -> ~10 dB, std=0.5 -> ~6 dB.
NOISE_STD = 0.3

# -------------------- LDPC channel coding (inference / eval only) --------------------
USE_LDPC = False            # toggle LDPC on/off; never active during training
LDPC_CODE_RATE = 0.5        # rate-1/2: doubles the number of transmitted bits
LDPC_MAX_ITER = 50          # belief-propagation decoding iterations

# -------------------- Training --------------------
BATCH_SIZE = 128
LR = 3e-4
EPOCHS = 50
EARLY_STOP_PATIENCE = 8
RECON_LOSS_WEIGHT = 5.0     # weight on AE reconstruction MSE (regularizer)
LABEL_SMOOTHING = 0.05
# Class weights for CrossEntropy (order matches CATEGORY_NAMES)
CLASS_WEIGHTS = [1.0, 1.0, 1.15, 1.15]


def snr_db_from_std(noise_std: float, power: float = QAM_POWER) -> float:
    """Convert an AWGN std to an approximate SNR in dB for the given symbol power."""
    import math
    if noise_std <= 0:
        return float("inf")
    return 10.0 * math.log10(power / (noise_std ** 2))
