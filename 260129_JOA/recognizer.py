# sign2talk/recognizer.py
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import copy
import tensorflow as tf

from .config import (
    TARGET_FRAMES, FEATURE_DIM, CONFIDENCE_THRESHOLD,
    CAM_INDEX, CAM_W, CAM_H,
    SLIM_V1_MODEL_PATH, SLIM_V2_MODEL_PATH, SLIM_V3_MODEL_PATH,
    SLIM_V3_MEAN_PATH, SLIM_V3_STD_PATH, SLIM_V3_CLASSNAMES_PATH,
)

from .labels import KOR_MAIN, ENG_MAIN
from .features import extract_frame_features, motion_score
from .preprocess import resample_to_fixed_length
from .ui import draw_panel, put_korean_text, wrap_text_by_chars, get_fonts
from .translator import EnKoTranslator, make_final_korean_sentence

from .recognizer_engine import ConsecutiveCommitter, ConsecutiveCommitConfig

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# =========================
# ✅ 성능 최적화(렉 감소)
# =========================
PROCESS_EVERY_N = 2
DRAW_LANDMARKS = True
HIDE_DEBUG_LINE_UNDER_LIVE = True

# =========================
# ✅ 랜드마크 깜빡임 방지
# =========================
LANDMARK_HOLD_FRAMES = 8

# =========================
# ✅ 문장 FINALIZE 정책
# =========================
AUTO_FINALIZE_ON_IDLE = True
IDLE_FINALIZE_SEC = 3.0
IDLE_MOTION_TH = 0.008

# =========================
# ✅ 오탐 방어(전이 구간)
# =========================
# 커밋 직후/세그먼트 종료 직후 잠깐 "잠금" -> transition 구간 이상 단어 방지
TRANSITION_LOCK_SEC = 0.30  # 0.25~0.35 추천

# 움직임이 너무 작으면(전이/정지) 커밋 금지
MIN_MOTION_FOR_COMMIT = 0.012  # START_TH보다 살짝 큰 값 추천

# =========================
# ✅ threshold / margin
# =========================
REJECT_IF_TOP1_BELOW = 0.50
REJECT_IF_MARGIN_BELOW = 0.07   # ✅ 전이 오탐 줄이려고 margin 조금 강화(0.05 -> 0.07)

# =========================
# ✅ 빠른 동작 대응
# =========================
PRED_INTERVAL_SEC = 0.12
EMA_ALPHA = 0.65
ENSEMBLE_WEIGHTS = [0.3, 0.5, 0.2]

# =========================
# ✅ 세그먼트 파라미터
# =========================
START_TH = 0.010
END_TH = 0.006
END_HOLD = 7
MIN_SEG_FRAMES = 6
MAX_SEG_FRAMES = 70
COOLDOWN_FRAMES = 6

# =========================
# ✅ 연속 N번 인식 커밋
# =========================
REQUIRED_HITS = 3
MAX_GAP_SEC = 0.40
COOLDOWN_SEC = 0.35

# ✅ 애매하면(마진 작으면) hits 더 요구
LOW_MARGIN_EXTRA_HITS = True
LOW_MARGIN_TH = 0.10           # margin < 0.10 이면 애매한 상태
LOW_MARGIN_REQUIRED_HITS = 4   # 이때는 4번 연속이어야 커밋

DISABLE_CONSECUTIVE_DUPLICATE = True

# =========================
# ✅ 특정 단어는 더 엄격하게
# =========================
STRICT_CLASSES = {
    "아직": {"min_prob": 0.88, "min_margin": 0.18},
}
CONFUSION_OVERRIDES = {
    ("아직", "만나다"): 0.06,
}


def compute_diffs(x):
    v = x[:, 1:, :] - x[:, :-1, :]
    v = tf.pad(v, [[0, 0], [1, 0], [0, 0]])
    a = v[:, 1:, :] - v[:, :-1, :]
    a = tf.pad(a, [[0, 0], [1, 0], [0, 0]])
    return tf.concat([x, v, a], axis=-1)


def load_ensemble_assets():
    with open(SLIM_V3_CLASSNAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    mean = np.load(SLIM_V3_MEAN_PATH)
    std = np.load(SLIM_V3_STD_PATH)

    print("🚀 앙상블 모델 로드 중...")
    m1 = tf.keras.models.load_model(
        SLIM_V1_MODEL_PATH,
        custom_objects={"compute_diffs": compute_diffs},
        compile=False
    )
    m2 = tf.keras.models.load_model(SLIM_V2_MODEL_PATH, compile=False)
    m3 = tf.keras.models.load_model(SLIM_V3_MODEL_PATH, compile=False)

    print("✅ slim_v1/v2/v3 로드 완료")
    print("✅ mean/std/class_names 로드 완료")
    return [m1, m2, m3], mean, std, class_names


def _finalize_and_clear_words(translator, sentence_eng, sentence_kor):
    if len(sentence_eng) > 0:
        final_eng_sent, final_kor_sent = make_final_korean_sentence(
            translator, sentence_eng, sentence_kor
        )
    else:
        final_eng_sent, final_kor_sent = "", "(단어가 없어서 문장 확정 불가)"
    sentence_eng.clear()
    sentence_kor.clear()
    return final_eng_sent, final_kor_sent


def _apply_confusion_override(top1_kor, top2_kor, margin):
    th = CONFUSION_OVERRIDES.get((top1_kor, top2_kor), None)
    if th is None:
        return False
    return margin < th


def _passes_commit_gate(word_kor, prob, margin):
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


def _map_code_to_labels(code_str: str):
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


def predict_ensemble(models, fixed30_feats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.expand_dims(fixed30_feats, axis=0).astype(np.float32)  # (1,30,47)
    x = (x - mean) / (std + 1e-8)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

    p1 = models[0](x_tf, training=False)[0].numpy()
    p2 = models[1](x_tf, training=False)[0].numpy()
    p3 = models[2](x_tf, training=False)[0].numpy()

    w = ENSEMBLE_WEIGHTS
    return p1 * w[0] + p2 * w[1] + p3 * w[2]


def main():
    models, MEAN, STD, CLASS_NAMES = load_ensemble_assets()
    _FONT_MAIN, _FONT_SENT = get_fonts()
    translator = EnKoTranslator()

    committer = ConsecutiveCommitter(ConsecutiveCommitConfig(
        required_hits=REQUIRED_HITS,
        max_gap_sec=MAX_GAP_SEC,
        cooldown_sec=COOLDOWN_SEC,
        min_prob=REJECT_IF_TOP1_BELOW,
        disable_consecutive_duplicate=DISABLE_CONSECUTIVE_DUPLICATE,
    ))

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다!")
        return

    cv2.namedWindow("Camera (Live only)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sentence", cv2.WINDOW_NORMAL)

    is_recording = False
    feats_prev = None
    motion_ema = 0.0

    in_segment = False
    end_hold_cnt = 0
    segment_feats = []

    ema_probs = None
    cooldown_frames = 0
    latest_live_text = "-"

    sentence_kor = []
    sentence_eng = []
    last_committed_kor = None

    finalized_once = False
    final_eng_sent = ""
    final_kor_sent = ""

    last_pred_time = 0.0
    idle_start_t = None

    # ✅ 전이 잠금 타이머
    lock_until_t = 0.0

    # 랜드마크 캐시
    cached_left = None
    cached_right = None
    cached_pose = None
    miss_left = 0
    miss_right = 0
    miss_pose = 0

    frame_idx = 0
    last_results = None

    print("=" * 70)
    print("✅ recognizer.py [Acc+ Transition-robust]")
    print(" - Space: REC | 0: Finalize | r: reset | q: quit")
    print(f" - TH={REJECT_IF_TOP1_BELOW:.2f} | margin>={REJECT_IF_MARGIN_BELOW:.2f}")
    print(f" - TransitionLock={TRANSITION_LOCK_SEC:.2f}s | MinMotionForCommit={MIN_MOTION_FOR_COMMIT:.3f}")
    print("=" * 70)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        smooth_landmarks=True,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_idx += 1
            now_t = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MP 스킵/재사용
            if in_segment:
                results = holistic.process(rgb)
                last_results = results
            else:
                if (frame_idx % PROCESS_EVERY_N == 0) or (last_results is None):
                    results = holistic.process(rgb)
                    last_results = results
                else:
                    results = last_results

            # 랜드마크 hold
            if results.left_hand_landmarks is not None:
                cached_left = copy.deepcopy(results.left_hand_landmarks)
                miss_left = 0
            else:
                miss_left += 1
                if cached_left is not None and miss_left <= LANDMARK_HOLD_FRAMES:
                    results.left_hand_landmarks = cached_left

            if results.right_hand_landmarks is not None:
                cached_right = copy.deepcopy(results.right_hand_landmarks)
                miss_right = 0
            else:
                miss_right += 1
                if cached_right is not None and miss_right <= LANDMARK_HOLD_FRAMES:
                    results.right_hand_landmarks = cached_right

            if results.pose_landmarks is not None:
                cached_pose = copy.deepcopy(results.pose_landmarks)
                miss_pose = 0
            else:
                miss_pose += 1
                if cached_pose is not None and miss_pose <= LANDMARK_HOLD_FRAMES:
                    results.pose_landmarks = cached_pose

            # draw
            if DRAW_LANDMARKS:
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if cooldown_frames > 0:
                cooldown_frames -= 1

            # -------------------------
            # 인식
            # -------------------------
            if is_recording:
                feats_now = extract_frame_features(results, feature_dim=FEATURE_DIM)
                m = motion_score(feats_now, feats_prev)
                feats_prev = feats_now
                motion_ema = 0.7 * motion_ema + 0.3 * m

                # ✅ IDLE 3초 FINALIZE
                if AUTO_FINALIZE_ON_IDLE and (len(sentence_eng) > 0) and (not finalized_once):
                    is_idle = (motion_ema < IDLE_MOTION_TH)
                    if is_idle:
                        if idle_start_t is None:
                            idle_start_t = now_t
                        elif (now_t - idle_start_t) >= IDLE_FINALIZE_SEC:
                            final_eng_sent, final_kor_sent = _finalize_and_clear_words(
                                translator, sentence_eng, sentence_kor
                            )
                            finalized_once = True
                            last_committed_kor = None
                            committer.reset()

                            # ✅ 강제 초기화 + 전이 잠금
                            in_segment = False
                            end_hold_cnt = 0
                            segment_feats = []
                            ema_probs = None
                            latest_live_text = "-"
                            cooldown_frames = 0
                            last_pred_time = 0.0
                            idle_start_t = None
                            lock_until_t = now_t + TRANSITION_LOCK_SEC
                    else:
                        idle_start_t = None
                else:
                    idle_start_t = None

                # 세그먼트 시작
                if (not in_segment) and (motion_ema >= START_TH):
                    in_segment = True
                    end_hold_cnt = 0
                    segment_feats = [feats_now]
                    ema_probs = None

                elif in_segment:
                    segment_feats.append(feats_now)

                    if motion_ema <= END_TH:
                        end_hold_cnt += 1
                    else:
                        end_hold_cnt = 0

                    too_long = (len(segment_feats) >= MAX_SEG_FRAMES)
                    allow_predict = (now_t - last_pred_time) >= PRED_INTERVAL_SEC

                    # ✅ 전이 잠금 동안은 예측은 하더라도 커밋 금지(=이상 단어 방지)
                    in_lock = (now_t < lock_until_t)

                    if allow_predict and len(segment_feats) >= MIN_SEG_FRAMES:
                        last_pred_time = now_t

                        seg = np.array(segment_feats, dtype=np.float32)
                        fixed30 = resample_to_fixed_length(seg, TARGET_FRAMES)
                        probs = predict_ensemble(models, fixed30, MEAN, STD)

                        if ema_probs is None:
                            ema_probs = probs.copy()
                        else:
                            ema_probs = (1 - EMA_ALPHA) * ema_probs + EMA_ALPHA * probs

                        order = np.argsort(ema_probs)[::-1]
                        top1, top2 = int(order[0]), int(order[1])
                        top1_p, top2_p = float(ema_probs[top1]), float(ema_probs[top2])
                        margin = top1_p - top2_p

                        code1 = str(CLASS_NAMES[top1])
                        code2 = str(CLASS_NAMES[top2])
                        top1_kor, top1_eng = _map_code_to_labels(code1)
                        top2_kor, top2_eng = _map_code_to_labels(code2)

                        use_top2 = _apply_confusion_override(top1_kor, top2_kor, margin)
                        chosen_kor = top2_kor if use_top2 else top1_kor
                        chosen_eng = top2_eng if use_top2 else top1_eng
                        chosen_p = top2_p if use_top2 else top1_p
                        chosen_margin = (top2_p - top1_p + 1e-6) if use_top2 else margin

                        latest_live_text = f"{code1} {chosen_kor} ({chosen_p*100:.1f}%)"

                        # ✅ 핵심: 전이/정지 구간 커밋 방지
                        can_try_commit = (
                            (cooldown_frames == 0)
                            and (not in_lock)
                            and (motion_ema >= MIN_MOTION_FOR_COMMIT)   # ✅ 움직임 너무 작으면 커밋 금지
                        )

                        if can_try_commit:
                            # ✅ 애매하면 hits 더 요구 (전이 구간 오탐 감소)
                            if LOW_MARGIN_EXTRA_HITS and (chosen_margin < LOW_MARGIN_TH):
                                committer.cfg.required_hits = LOW_MARGIN_REQUIRED_HITS
                            else:
                                committer.cfg.required_hits = REQUIRED_HITS

                            should_commit, last_committed_kor = committer.update_and_maybe_commit(
                                chosen_kor=chosen_kor,
                                chosen_eng=chosen_eng,
                                chosen_prob=chosen_p,
                                chosen_margin=chosen_margin,
                                last_committed_kor=last_committed_kor,
                                passes_gate_fn=_passes_commit_gate,
                                now_t=now_t,
                            )
                        else:
                            should_commit = False

                        if should_commit:
                            sentence_kor.append(chosen_kor)
                            sentence_eng.append(chosen_eng if chosen_eng is not None else "unknown")

                            finalized_once = False
                            final_eng_sent = ""
                            final_kor_sent = ""

                            cooldown_frames = COOLDOWN_FRAMES

                            # ✅ 커밋 직후 전이 잠금
                            lock_until_t = now_t + TRANSITION_LOCK_SEC

                            # 다음 단어로 넘어가기
                            in_segment = False
                            end_hold_cnt = 0
                            segment_feats = []
                            ema_probs = None
                            continue

                    # 세그먼트 종료(끝났을 때도 전이 잠금)
                    if end_hold_cnt >= END_HOLD or too_long:
                        in_segment = False
                        end_hold_cnt = 0
                        segment_feats = []
                        ema_probs = None
                        lock_until_t = now_t + TRANSITION_LOCK_SEC  # ✅ transition 보호

            # -------------------------
            # UI
            # -------------------------
            frame_ui = frame.copy()
            frame_ui = draw_panel(frame_ui, 15, 15, 1080, 210, alpha=0.72)

            if is_recording:
                cv2.circle(frame_ui, (35, 55), 7, (0, 0, 255), -1)
                frame_ui = put_korean_text(frame_ui, "REC", (55, 35),
                                           color=(255, 255, 255), font=_FONT_MAIN, stroke_width=3)
            else:
                cv2.circle(frame_ui, (35, 55), 7, (130, 130, 130), -1)
                frame_ui = put_korean_text(frame_ui, "IDLE", (55, 35),
                                           color=(220, 220, 220), font=_FONT_MAIN, stroke_width=3)

            live_show = latest_live_text if is_recording else "-"
            frame_ui = put_korean_text(
                frame_ui,
                f"LIVE: {live_show}",
                (25, 70),
                color=(0, 255, 255),
                font=_FONT_MAIN,
                stroke_width=4
            )

            if not HIDE_DEBUG_LINE_UNDER_LIVE:
                lock_left = max(0.0, lock_until_t - time.time())
                frame_ui = put_korean_text(
                    frame_ui,
                    f"motion_ema={motion_ema:.4f} | in_seg={in_segment} | lock_left={lock_left:.2f}s | "
                    f"cooldown_frames={cooldown_frames} | pred_int={PRED_INTERVAL_SEC:.2f}s",
                    (25, 160),
                    color=(200, 200, 200),
                    font=_FONT_MAIN,
                    stroke_width=2
                )

            hint = "Space: REC | 0: Finalize | R: Reset | Q: Quit"
            frame_ui = put_korean_text(
                frame_ui,
                hint,
                (15, frame_ui.shape[0] - 45),
                color=(255, 255, 255),
                font=_FONT_MAIN,
                stroke_width=3
            )
            cv2.imshow("Camera (Live only)", frame_ui)

            sent_canvas = np.zeros((420, 1200, 3), dtype=np.uint8)
            sent_canvas = draw_panel(sent_canvas, 0, 0, sent_canvas.shape[1], sent_canvas.shape[0], alpha=1.0)

            sent_canvas = put_korean_text(
                sent_canvas,
                "SENTENCE (CAM) [Transition-robust]",
                (20, 15),
                color=(200, 200, 200),
                font=_FONT_MAIN,
                stroke_width=3
            )

            sent_text = " ".join(sentence_kor) if sentence_kor else "(아직 커밋된 단어 없음)"
            lines = wrap_text_by_chars(sent_text, max_chars=28)[:2]
            y0 = 85
            for i, line in enumerate(lines):
                sent_canvas = put_korean_text(
                    sent_canvas,
                    line,
                    (20, y0 + i * 75),
                    color=(0, 255, 0),
                    font=_FONT_SENT,
                    stroke_width=6
                )

            sub = (
                f"TH={REJECT_IF_TOP1_BELOW:.2f} | margin>={REJECT_IF_MARGIN_BELOW:.2f} | "
                f"hits={REQUIRED_HITS}/{LOW_MARGIN_REQUIRED_HITS}(low-margin) | Words={len(sentence_kor)}"
            )
            sent_canvas = put_korean_text(
                sent_canvas,
                sub,
                (20, 235),
                color=(180, 180, 180),
                font=_FONT_MAIN,
                stroke_width=2
            )

            sent_canvas = put_korean_text(
                sent_canvas,
                "FINAL (KO):",
                (20, 275),
                color=(200, 200, 200),
                font=_FONT_MAIN,
                stroke_width=2
            )

            final_show = final_kor_sent if final_kor_sent.strip() else "(아직 문장 확정 없음) - 3초 정지 시 자동 생성"
            sent_canvas = put_korean_text(
                sent_canvas,
                final_show,
                (20, 315),
                color=(255, 255, 255),
                font=_FONT_MAIN,
                stroke_width=3
            )

            if final_eng_sent.strip():
                sent_canvas = put_korean_text(
                    sent_canvas,
                    f"EN: {final_eng_sent}",
                    (20, 365),
                    color=(160, 160, 160),
                    font=_FONT_MAIN,
                    stroke_width=2
                )

            cv2.imshow("Sentence", sent_canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                is_recording = not is_recording
                if is_recording:
                    feats_prev = None
                    motion_ema = 0.0

                    in_segment = False
                    end_hold_cnt = 0
                    segment_feats = []
                    ema_probs = None

                    cooldown_frames = 0
                    latest_live_text = "CAM INPUT MODE"

                    sentence_kor.clear()
                    sentence_eng.clear()
                    last_committed_kor = None

                    finalized_once = False
                    final_eng_sent = ""
                    final_kor_sent = ""

                    last_pred_time = 0.0
                    idle_start_t = None
                    lock_until_t = 0.0
                    committer.reset()

                    print("\n* REC ON")
                else:
                    latest_live_text = "-"
                    print("* REC OFF")

            elif key == ord("r"):
                sentence_kor.clear()
                sentence_eng.clear()
                last_committed_kor = None

                finalized_once = False
                final_eng_sent = ""
                final_kor_sent = ""
                latest_live_text = "-"

                in_segment = False
                end_hold_cnt = 0
                segment_feats = []
                ema_probs = None

                cooldown_frames = 0
                last_pred_time = 0.0
                idle_start_t = None
                lock_until_t = 0.0
                committer.reset()

                print("[RESET] cleared")

            elif key == ord("0"):
                final_eng_sent, final_kor_sent = _finalize_and_clear_words(
                    translator, sentence_eng, sentence_kor
                )
                finalized_once = True
                last_committed_kor = None

                in_segment = False
                end_hold_cnt = 0
                segment_feats = []
                ema_probs = None

                cooldown_frames = 0
                latest_live_text = "-"
                last_pred_time = 0.0
                idle_start_t = None
                lock_until_t = time.time() + TRANSITION_LOCK_SEC
                committer.reset()

                print(f"[FINALIZE-MANUAL] EN={final_eng_sent} | KO={final_kor_sent}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n종료")
