# sign2talk/config.py
import os

# =========================================================
# Global constants
# =========================================================
TARGET_FRAMES = 30
FEATURE_DIM = 47

# ✅ sentence로 "커밋"될 최소 confidence
CONFIDENCE_THRESHOLD = 0.70

# =========================================================
# (기존) Transformer 모델 경로 (유지)
# =========================================================
MODEL_NAME = "Transformer_v1"

BASE_DIR = r"C:\Users\hello\Desktop\Sign2Talk\sign2talk"
WEIGHTS_PATH    = fr"{BASE_DIR}\{MODEL_NAME}.weights.h5"
PREPROCESS_PATH = fr"{BASE_DIR}\{MODEL_NAME}_preprocess.npz"

# =========================================================
# ✅ (추가) slim_v1/v2/v3 앙상블 모델/전처리 경로
# - 친구 코드: slim_v1.keras, slim_v2.keras, slim_v3.keras
# - 전처리: mean.npy, std.npy, class_names.json
# =========================================================
SLIM_V1_MODEL_PATH      = fr"{BASE_DIR}\slim_v1.keras"
SLIM_V2_MODEL_PATH      = fr"{BASE_DIR}\slim_v2.keras"
SLIM_V3_MODEL_PATH      = fr"{BASE_DIR}\slim_v3.keras"

SLIM_V3_MEAN_PATH       = fr"{BASE_DIR}\mean.npy"
SLIM_V3_STD_PATH        = fr"{BASE_DIR}\std.npy"
SLIM_V3_CLASSNAMES_PATH = fr"{BASE_DIR}\class_names.json"

# =========================================================
# Font
# =========================================================
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_SIZE_MAIN = 32
FONT_SIZE_SENT = 44

if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"폰트 파일을 찾을 수 없습니다: {FONT_PATH}")

# =========================================================
# Webcam settings
# =========================================================
CAM_INDEX = 0
CAM_W = 1280
CAM_H = 720

# =========================================================
# (선택) 모델 파일 존재 체크(에러 빨리 찾기)
# =========================================================
for p in [
    SLIM_V1_MODEL_PATH, SLIM_V2_MODEL_PATH, SLIM_V3_MODEL_PATH,
    SLIM_V3_MEAN_PATH, SLIM_V3_STD_PATH, SLIM_V3_CLASSNAMES_PATH
]:
    if not os.path.exists(p):
        print(f"⚠️ 파일이 없습니다: {p}")
