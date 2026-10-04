"""Central constants. Column names follow the UCI AI4I 2020 file (ai4i2020.csv)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "ai4i2020.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "rf_engineered.joblib"

TARGET = "Machine failure"
MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]   # LEAKAGE: never use as model inputs
ID_COLS = ["UDI", "Product ID"]                    # identifiers: never used as features

AIR = "Air temperature [K]"
PROC = "Process temperature [K]"
RPM = "Rotational speed [rpm]"
TORQUE = "Torque [Nm]"
WEAR = "Tool wear [min]"
TYPE = "Type"

SENSOR_COLS = [AIR, PROC, RPM, TORQUE, WEAR]
REQUIRED_INPUT_COLS = SENSOR_COLS + [TYPE]

TYPE_MAP = {"L": 0, "M": 1, "H": 2}
STRAIN_LIMIT = {"L": 11000, "M": 12000, "H": 13000}  # minNm, from the dataset documentation

# Documented failure rules (UCI dataset description)
HDF_MAX_DIFF = 8.6      # K   (process - air temperature)
HDF_MAX_RPM = 1380      # rpm
PWF_MIN_W = 3500.0      # W
PWF_MAX_W = 9000.0      # W

# Broad physical-plausibility limits used only to reject garbage input rows
PLAUSIBLE = {
    AIR: (250.0, 400.0),
    PROC: (250.0, 450.0),
    RPM: (1.0, 10000.0),
    TORQUE: (0.0, 500.0),
    WEAR: (0.0, 1000.0),
}

RANDOM_STATE = 42
PRODUCTION_THRESHOLD = 0.515
