import os

# =========================================================
# Global constants (원본 그대로)
# =========================================================
TARGET_FRAMES = 30
FEATURE_DIM = 47

# ✅ sentence로 "커밋"될 최소 confidence
CONFIDENCE_THRESHOLD = 0.70

# =========================================================
# Model / Preprocess paths (원본 그대로)
# =========================================================
MODEL_NAME = "Transformer_v1"

BASE_DIR = r"C:\Users\hello\Desktop\Sign2Talk\sign2talk"
WEIGHTS_PATH    = fr"{BASE_DIR}\{MODEL_NAME}.weights.h5"
PREPROCESS_PATH = fr"{BASE_DIR}\{MODEL_NAME}_preprocess.npz"

# =========================================================
# Font (원본 그대로)
# =========================================================
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_SIZE_MAIN = 32
FONT_SIZE_SENT = 44

if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"폰트 파일을 찾을 수 없습니다: {FONT_PATH}")

# =========================================================
# Webcam settings (원본 그대로)
# =========================================================
CAM_INDEX = 0
CAM_W = 1280
CAM_H = 720
