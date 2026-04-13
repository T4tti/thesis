import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from datetime import datetime
import torch

# Reduce TensorFlow/XLA startup logs in notebook output
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_CPP_MIN_LOG_LEVEL", "3")

# Default to CPU to avoid CUDA plugin registration noise.
# Set TIMEGAN_ENABLE_GPU=1 before running notebook to enable GPU.
os.environ.setdefault("TIMEGAN_ENABLE_GPU", "0")
if os.environ.get("TIMEGAN_ENABLE_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# TimeGAN imports (tsgm)
TIMEGAN_AVAILABLE = True
TIMEGAN_IMPORT_ERROR = None
try:
    import tsgm
    from tsgm.models.timeGAN import TimeGAN
except Exception as e:
    tsgm = None
    TimeGAN = None
    TIMEGAN_AVAILABLE = False
    TIMEGAN_IMPORT_ERROR = str(e)

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
    tf.get_logger().setLevel("ERROR")
except Exception:
    tf = None

# Display settings
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')

print("Libraries imported successfully")
print(f"Random seed: {RANDOM_SEED}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available (PyTorch): {torch.cuda.is_available()}")
print(f"TensorFlow available: {tf is not None}")
print(f"TimeGAN import available: {TIMEGAN_AVAILABLE}")
print(f"TIMEGAN_ENABLE_GPU={os.environ.get('TIMEGAN_ENABLE_GPU', '0')}")
if TIMEGAN_AVAILABLE and tsgm is not None:
    print(f"TSGM version: {getattr(tsgm, '__version__', 'unknown')}")
if not TIMEGAN_AVAILABLE:
    print(f"TimeGAN import error: {TIMEGAN_IMPORT_ERROR}")