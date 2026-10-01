# sign2talk/recognizer_utils.py
import json
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from .config import (
    SLIM_V1_MODEL_PATH,
    SLIM_V2_MODEL_PATH,
    SLIM_V3_MODEL_PATH,
    SLIM_V3_MEAN_PATH,
    SLIM_V3_STD_PATH,
    SLIM_V3_CLASSNAMES_PATH,
)
from .labels import KOR_MAIN, ENG_MAIN
from .translator import EnKoTranslator, make_final_korean_sentence
from .recognizer_tuning import (
    ENSEMBLE_WEIGHTS,
    CONFUSION_OVERRIDES,
    FORCE_TOP2_IF_TOP1,      # ✅ 추가
    STRICT_CLASSES,
    REJECT_IF_TOP1_BELOW,
    REJECT_IF_MARGIN_BELOW,
)


def compute_diffs(x: tf.Tensor) -> tf.Tensor:
    """입력 시퀀스에 1차/2차 차분을 붙여주는 helper (모델 custom_objects)."""
    v = x[:, 1:, :] - x[:, :-1, :]
    v = tf.pad(v, [[0, 0], [1, 0], [0, 0]])
    a = v[:, 1:, :] - v[:, :-1, :]
    a = tf.pad(a, [[0, 0], [1, 0], [0, 0]])
    return tf.concat([x, v, a], axis=-1)


def load_ensemble_assets():
    """앙상블 모델 + 정규화 정보 + 클래스 이름 로드."""
    with open(SLIM_V3_CLASSNAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    mean = np.load(SLIM_V3_MEAN_PATH)
    std = np.load(SLIM_V3_STD_PATH)

    print("🚀 앙상블 모델 로드 중...")
    m1 = tf.keras.models.load_model(
        SLIM_V1_MODEL_PATH,
        custom_objects={"compute_diffs": compute_diffs},
        compile=False,
    )
    m2 = tf.keras.models.load_model(SLIM_V2_MODEL_PATH, compile=False)
    m3 = tf.keras.models.load_model(SLIM_V3_MODEL_PATH, compile=False)

    print("✅ slim_v1/v2/v3 로드 완료")
    print("✅ mean/std/class_names 로드 완료")
    return [m1, m2, m3], mean, std, class_names


def finalize_and_clear_words(
    translator: EnKoTranslator,
    sentence_eng: List[str],
    sentence_kor: List[str],
) -> Tuple[str, str]:
    """현재까지 인식된 단어 리스트로 최종 문장 생성 후 리스트를 비워준다."""
    if len(sentence_eng) > 0:
        final_eng_sent, final_kor_sent = make_final_korean_sentence(
            translator, sentence_eng, sentence_kor
        )
    else:
        final_eng_sent, final_kor_sent = "", "(단어가 없어서 문장 확정 불가)"

    sentence_eng.clear()
    sentence_kor.clear()
    return final_eng_sent, final_kor_sent


def apply_confusion_override(top1_kor: str, top2_kor: str, margin: float) -> bool:
    """
    특정 혼동 규칙에 따라 top2를 선택할지 여부.
    True를 반환하면 top2를 선택한다.
    """
    # 1) (top1, top2) 특정 쌍 규칙
    th = CONFUSION_OVERRIDES.get((top1_kor, top2_kor), None)
    if th is not None:
        return margin < th

    # 2) top1이 특정 단어이면(예: 인간), margin이 작을 때는 top2로 강제
    th2 = FORCE_TOP2_IF_TOP1.get(top1_kor, None)
    if th2 is not None:
        return margin < th2

    return False


def passes_commit_gate(word_kor: str, prob: float, margin: float) -> bool:
    """확률/마진 및 class별 규칙을 만족하는지 여부."""
    if prob < REJECT_IF_TOP1_BELOW:
        return False
    if margin < REJECT_IF_MARGIN_BELOW:
        return False

    if word_kor in STRICT_CLASSES:
        rule = STRICT_CLASSES[word_kor]
        if prob < rule["min_prob"]:
            return False
        if margin < rule["min_margin"]:
            return False

    return True


def map_code_to_labels(code_str: str) -> Tuple[str, str]:
    """클래스 코드 문자열을 한/영 label로 매핑."""
    try:
        code_int = int(code_str)
    except Exception:
        code_int = None

    kor = KOR_MAIN.get(code_int, None) if code_int is not None else None
    eng = ENG_MAIN.get(code_int, None) if code_int is not None else None

    if kor is None:
        kor = f"CODE_{code_str}"
    if eng is None:
        eng = f"code_{code_str}"
    return kor, eng


def predict_ensemble(
    models,
    fixed30_feats: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """앙상블 모델로부터 softmax 평균을 계산."""
    x = np.expand_dims(fixed30_feats, axis=0).astype(np.float32)  # (1,30,47)
    x = (x - mean) / (std + 1e-8)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

    p1 = models[0](x_tf, training=False)[0].numpy()
    p2 = models[1](x_tf, training=False)[0].numpy()
    p3 = models[2](x_tf, training=False)[0].numpy()

    w = ENSEMBLE_WEIGHTS
    return p1 * w[0] + p2 * w[1] + p3 * w[2]
