# sign2talk/recognizer_tuning.py
import numpy as np

# =========================
# ✅ 성능 최적화(렉 감소)
# =========================
PROCESS_EVERY_N = 3
DRAW_LANDMARKS = True
HIDE_DEBUG_LINE_UNDER_LIVE = True

# =========================
# ✅ 랜드마크 깜빡임 방지
# =========================
LANDMARK_HOLD_FRAMES = 6

# =========================
# ✅ 문장 FINALIZE 정책
# =========================
AUTO_FINALIZE_ON_IDLE = True
IDLE_FINALIZE_SEC = 3.0
IDLE_MOTION_TH = 0.008

# =========================
# ✅ 오탐 방어(전이 구간)
# =========================
TRANSITION_LOCK_SEC = 0.12
MIN_MOTION_FOR_COMMIT = 0.006

# =========================
# ✅ threshold / margin (하드 커밋 게이트)
# =========================
REJECT_IF_TOP1_BELOW = 0.45
REJECT_IF_MARGIN_BELOW = 0.05

# =========================
# ✅ 빠른 동작 대응
# =========================
PRED_INTERVAL_SEC = 0.10
EMA_ALPHA = 0.65
ENSEMBLE_WEIGHTS = [0.3, 0.5, 0.2]

# =========================
# ✅ 세그먼트 파라미터
# =========================
START_TH = 0.012
END_TH = 0.010
END_HOLD = 3
MIN_SEG_FRAMES = 6
MAX_SEG_FRAMES = 45
COOLDOWN_FRAMES = 4

# =========================
# ✅ 연속 N번 인식 커밋(legacy - 현재 best-of-segment 중심)
# =========================
REQUIRED_HITS = 3
MAX_GAP_SEC = 0.40
COOLDOWN_SEC = 0.35

LOW_MARGIN_EXTRA_HITS = True
LOW_MARGIN_TH = 0.10
LOW_MARGIN_REQUIRED_HITS = 4

# ✅ 문장 리스트에 연속 같은 단어 못 들어가게
DISABLE_CONSECUTIVE_DUPLICATE = True

# =========================
# ✅ 특정 단어는 더 엄격하게 (혼동 많이 나는 것들)
# =========================
STRICT_CLASSES = {
    # 아직(완화)
    "아직": {"min_prob": 0.55, "min_margin": 0.05},

    # ✅ 핵심: 인간이 너무 잘 뜨는 문제 -> "인간"을 더 엄격하게 막아서
    # 인간이 애매한 상황에서는 top1으로 붙는 걸 못 하게 하고,
    # 아래 FORCE_TOP2_IF_TOP1 규칙이 함께 작동하면 만나다로 넘어갈 확률이 커짐
    "인간": {"min_prob": 0.82, "min_margin": 0.18},

    # 땀은 기존 유지(원하면 인간처럼 더 올려도 됨)
    "땀": {"min_prob": 0.70, "min_margin": 0.12},
}

# =========================
# ✅ 혼동쌍 override (top1/top2 간격이 애매할 때 보정)
# =========================
CONFUSION_OVERRIDES = {
    ("아직", "만나다"): 0.06,

    # 인간이 top1이고 만나다가 top2로 올라온 경우에만 뒤집을 수 있음
    ("인간", "만나다"): 0.10,

    ("땀", "보다"): 0.10,
}

# =========================
# ✅ 추가 규칙: "top1=인간"이면 margin이 애매할 때 top2로 강제
# =========================
# - top2가 만나다가 아니어도, 인간 고정을 깨는 데 효과적
# - 만나다가 top2에 자주 올라오기만 하면, 인간 대신 만나다가 선택될 확률이 확 올라감
FORCE_TOP2_IF_TOP1 = {
    "인간": 0.18,  # margin < 0.18 이면 top1(인간) 버리고 top2 선택
}

FAST_CLASSES = {
    "나": {
        "min_prob": 0.55,
        "min_margin": 0.02,
        "min_duration": 0.6,  # 초
    }
}

# =========================
# ✅ 소프트 커밋(확률 낮아도 2~3번이면 커밋)
# =========================
SOFT_COMMIT_MIN_PROB = 0.40
SOFT_COMMIT_REQUIRED = 3
SOFT_COMMIT_WINDOW_SEC = 1.5
HARD_COMMIT_PROB = 0.75
