"""
Main GLobal Config file for the whole project/experiments. Although there is some rules about it


Professional ML libraries like HuggingFace or Pytorch maintain these standards about config:

1. Core library must NEVER depend on config.py
2. Core functions should NEVER hardcode paths

"""


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"

# checking if the folders exist
for p in [RESULTS_ROOT, CHECKPOINT_ROOT]:
    p.mkdir(exist_ok=True, parents=True)


# dataset registry
DEFAULT_DATASET = "indian_pines"

# CV and Training Parameter Defaults
DEFAULT_PATCH_SIZE = 25
DEFAULT_SMALL_PATCH = 5

DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS     = 50
DEFAULT_FOLDS      = 10

# SSL settings
DEFAULT_SSL_ITERS     = 5
DEFAULT_SSL_THRESHOLD = 0.90
DEFAULT_SSL_LABELED_FRAC = 0.05

# PCA
PCA_VARIANCE_THRESHOLD = 0.99
PCA_MAX_COMPONENTS     = 30

# Random seed (for numpy / tf / sklearn)
GLOBAL_SEED = 0


# Band Selection Defaults
DEFAULT_NUM_BANDS = 10
IBRA_VIF_THRESHOLD = 10
GSS_MAX_DISTANCE = 5
SRPA_LAMBDA = 0.3
SSEP_TOPK = 20