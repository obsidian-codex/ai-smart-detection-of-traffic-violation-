"""Central configuration for Smart Traffic Intelligence System."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT.parent

# External dataset paths (sibling folders in MODELS/)
VISDRONE_TRAIN = DATASETS_ROOT / "VisDrone2019-VID-train"
VISDRONE_VAL = DATASETS_ROOT / "VisDrone2019-VID-val"
UA_DETRAC = DATASETS_ROOT / "ua-detrac"
HELMET_RAW = DATASETS_ROOT / "helmet detection"

# Project internal paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
HEATMAPS_DIR = OUTPUTS_DIR / "heatmaps"
SCREENSHOTS_DIR = OUTPUTS_DIR / "screenshots"

HELMET_DATASET_YOLO = DATASETS_DIR / "helmet_yolo"
HELMET_MODEL_PATH = MODELS_DIR / "helmet_yolo11s" / "weights" / "best.pt"
VEHICLE_MODEL = "yolo11s.pt"

# COCO class IDs for vehicles (YOLO11 pretrained)
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASS_NAMES = list(VEHICLE_CLASS_IDS.values())
# Single-pass YOLO: person + all vehicle classes
DETECTION_CLASS_IDS = {PERSON_CLASS_ID: "person", **VEHICLE_CLASS_IDS}

# Tripling detection (motorcycle > 2 riders)
TRIPLING_BBOX_MARGIN = 0.15
TRIPLING_MIN_RIDERS = 3

# Helmet classes
HELMET_CLASSES = ["helmet", "no_helmet"]

# Tracking
TRACKER_CONFIG = "bytetrack.yaml"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Speed estimation (calibrate per camera)
PIXELS_PER_METER = 8.0
FPS_DEFAULT = 25.0
SPEED_LIMIT_KMH = 60.0

# Congestion thresholds
CONGESTION_LOW_MAX = 35
CONGESTION_MEDIUM_MAX = 65

# Wrong-way detection
EXPECTED_FLOW_DIRECTION = (0, -1)  # upward in image coords (configurable)
WRONG_WAY_MIN_DISPLACEMENT = 15
WRONG_WAY_MIN_FRAMES = 5

# Risk score weights
RISK_WEIGHTS = {
    "congestion": 0.30,
    "wrong_way": 0.20,
    "helmet": 0.15,
    "overspeed": 0.15,
    "tripling": 0.20,
}

# Demo defaults
DEFAULT_DEMO_SEQUENCE = "uav0000071_03240_v"
