# sign2talk/recognizer_cam_v2.py (예: 기존 카메라 버전 파일을 이 코드로 교체)
import cv2
import mediapipe as mp
import numpy as np
import time

from .config import (
    TARGET_FRAMES, FEATURE_DIM, CONFIDENCE_THRESHOLD,
    MODEL_NAME, WEIGHTS_PATH, PREPROCESS_PATH,
    CAM_INDEX, CAM_W, CAM_H
)
from .models import build_model_by_name
from .labels import KOR_MAIN, ACTIVE_LABELS, ENG_MAIN
from .features import extract_frame_features, motion_score
from .preprocess import resample_to_fixed_length, predict_on_fixed30
from .ui import draw_panel, put_korean_text, wrap_text_by_chars, get_fonts
from .translator import EnKoTranslator, make_final_korean_sentence

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# =========================
# ✅ 카메라 버전도 키보드 v2와 동일한 문장 종료 정책
# =========================
SENTENCE_GAP_SEC = 1.5  # 자동 종료 쓰고 싶으면 >0 유지, 0키 수동 종료만 쓰려면 0.0 추천


def load_preprocess(npz_path: str):
    preproc = np.load(npz_path, allow_pickle=True)
    mean = preproc["mean"]
    std = preproc["std"]
    class_names = preproc["class_names"].tolist()
    return mean, std, class_names


def load_model():
    mean, std, class_names = load_preprocess(PREPROCESS_PATH)
    num_classes = len(class_names)

    # 라벨 일치 검증
    assert len(ACTIVE_LABELS) == num_classes

    input_shape = (TARGET_FRAMES, FEATURE_DIM)
    model = build_model_by_name(MODEL_NAME, input_shape, num_classes)
    model.load_weights(WEIGHTS_PATH)
    print("✅ 모델 구조 생성 + 가중치 로드 완료:", MODEL_NAME)

    return model, mean, std, class_names


def main():
    model, MEAN, STD, CLASS_NAMES = load_model()
    _FONT_MAIN, _FONT_SENT = get_fonts()

    # ✅ 번역 모델은 1번만 로드
    translator = EnKoTranslator()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다!")
        return

    # ✅ 창을 마우스로 마음대로 리사이즈 가능하게 (키보드 v2와 동일)
    cv2.namedWindow("Camera (Live only)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sentence", cv2.WINDOW_NORMAL)

    # =========================
    # 빠른 수어용 세그먼트 파라미터 (기존 유지)
    # =========================
    START_TH = 0.010
    END_TH = 0.008
    END_HOLD = 4

    MIN_SEG_FRAMES = 6
    MAX_SEG_FRAMES = 55

    # =========================
    # 커밋 안정 조건 (기존 유지)
    # =========================
    EMA_ALPHA = 0.55
    MARGIN_TH = 0.10
    COOLDOWN_FRAMES = 5

    # =========================
    # ✅ 조기 커밋(동작 중 커밋) 파라미터 (기존 유지)
    # =========================
    EARLY_PRED_EVERY = 1
    EARLY_MIN_FRAMES = 6
    EARLY_STABLE_N = 2
    EARLY_MARGIN_TH = 0.07

    # =========================
    # ✅ 키보드 v2와 동일한 상태 변수들로 통일
    # =========================
    is_recording = False

    # motion/segment
    feats_prev = None
    motion_ema = 0.0
    in_segment = False
    end_hold_cnt = 0
    segment_feats = []
    ema_probs = None
    cooldown = 0
    latest_live_text = "-"

    # 커밋된 단어들 (키보드 v2와 동일 네이밍)
    sentence_kor = []
    sentence_eng = []

    last_committed_kor = None
    last_commit_time = None

    # 문장 확정(번역 결과)
    finalized_once = False
    final_eng_sent = ""
    final_kor_sent = ""

    # 조기 안정화 상태
    early_stable_label = None
    early_stable_cnt = 0
    seg_frame_counter = 0

    print("=" * 70)
    print("✅ recognizer_cam_v2.py (WEBCAM) - keyboard v2와 UI/로직 통일")
    print(" - Space: Start/Stop (REC 토글)")
    print(" - 0: 문장 종료 -> 번역/검증 실행 (FINAL 출력)")
    print(" - r: reset")
    print(" - q: quit")
    print("=" * 70)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,  # 속도 우선
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("!!! 프레임을 읽을 수 없음")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # 랜드마크 draw
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if cooldown > 0:
                cooldown -= 1

            # =========================
            # 인식 ON일 때만
            # =========================
            if is_recording:
                feats_now = extract_frame_features(results, feature_dim=FEATURE_DIM)

                m = motion_score(feats_now, feats_prev)
                feats_prev = feats_now
                motion_ema = 0.7 * motion_ema + 0.3 * m

                # =========================
                # (선택) 자동 문장 종료 감지(침묵) → 번역
                # - 키보드 v2와 동일하게 옵션으로 유지
                # =========================
                if (
                    SENTENCE_GAP_SEC > 0
                    and (last_commit_time is not None)
                    and (not finalized_once)
                    and (len(sentence_eng) > 0)
                ):
                    if (time.time() - last_commit_time) >= SENTENCE_GAP_SEC:
                        final_eng_sent, final_kor_sent = make_final_korean_sentence(
                            translator, sentence_eng, sentence_kor
                        )
                        finalized_once = True

                # 세그먼트 시작
                if (not in_segment) and (motion_ema >= START_TH):
                    in_segment = True
                    end_hold_cnt = 0
                    segment_feats = [feats_now]
                    ema_probs = None

                    early_stable_label = None
                    early_stable_cnt = 0
                    seg_frame_counter = 0

                # 세그먼트 진행
                elif in_segment:
                    segment_feats.append(feats_now)
                    seg_frame_counter += 1

                    # ----- 조기 예측/조기 커밋 -----
                    if len(segment_feats) >= EARLY_MIN_FRAMES and (seg_frame_counter % EARLY_PRED_EVERY == 0):
                        seg = np.array(segment_feats, dtype=np.float32)
                        fixed30 = resample_to_fixed_length(seg, TARGET_FRAMES)
                        probs = predict_on_fixed30(model, fixed30, MEAN, STD)

                        if ema_probs is None:
                            ema_probs = probs.copy()
                        else:
                            ema_probs = (1 - EMA_ALPHA) * ema_probs + EMA_ALPHA * probs

                        order = np.argsort(ema_probs)[::-1]
                        top1, top2 = int(order[0]), int(order[1])
                        top1_p, top2_p = float(ema_probs[top1]), float(ema_probs[top2])
                        margin = top1_p - top2_p

                        top1_label_num = ACTIVE_LABELS[top1]
                        top1_kor = KOR_MAIN.get(top1_label_num, "UNKNOWN")
                        top1_eng = ENG_MAIN.get(top1_label_num, "unknown")

                        latest_live_text = f"{top1_label_num} {top1_kor} ({top1_p*100:.1f}%)"

                        if early_stable_label == top1:
                            early_stable_cnt += 1
                        else:
                            early_stable_label = top1
                            early_stable_cnt = 1

                        # ✅ 동작이 끝나기 전에도 바로 단어 확정
                        if (
                            cooldown == 0
                            and top1_p >= CONFIDENCE_THRESHOLD
                            and margin >= EARLY_MARGIN_TH
                            and early_stable_cnt >= EARLY_STABLE_N
                        ):
                            if last_committed_kor != top1_kor:
                                sentence_kor.append(top1_kor)
                                sentence_eng.append(top1_eng)
                                last_committed_kor = top1_kor

                                last_commit_time = time.time()
                                finalized_once = False
                                final_eng_sent = ""
                                final_kor_sent = ""

                                print(f"[EARLY COMMIT] {top1_label_num} {top1_kor} / {top1_eng}")

                            cooldown = COOLDOWN_FRAMES

                            # 다음 단어로: 세그먼트 강제 종료/리셋
                            in_segment = False
                            end_hold_cnt = 0
                            segment_feats = []
                            ema_probs = None
                            early_stable_label = None
                            early_stable_cnt = 0
                            seg_frame_counter = 0
                            continue

                    # ----- 종료 감지(멈춤) -----
                    if motion_ema <= END_TH:
                        end_hold_cnt += 1
                    else:
                        end_hold_cnt = 0

                    too_long = (len(segment_feats) >= MAX_SEG_FRAMES)

                    if end_hold_cnt >= END_HOLD or too_long:
                        seg_len = len(segment_feats)

                        in_segment = False
                        end_hold_cnt = 0

                        if seg_len >= MIN_SEG_FRAMES:
                            seg = np.array(segment_feats, dtype=np.float32)
                            fixed30 = resample_to_fixed_length(seg, TARGET_FRAMES)
                            probs = predict_on_fixed30(model, fixed30, MEAN, STD)

                            if ema_probs is None:
                                ema_probs = probs.copy()
                            else:
                                ema_probs = (1 - EMA_ALPHA) * ema_probs + EMA_ALPHA * probs

                            order = np.argsort(ema_probs)[::-1]
                            top1, top2 = int(order[0]), int(order[1])
                            top1_p, top2_p = float(ema_probs[top1]), float(ema_probs[top2])
                            margin = top1_p - top2_p

                            top1_label_num = ACTIVE_LABELS[top1]
                            top1_kor = KOR_MAIN.get(top1_label_num, "UNKNOWN")
                            top1_eng = ENG_MAIN.get(top1_label_num, "unknown")
                            latest_live_text = f"{top1_label_num} {top1_kor} ({top1_p*100:.1f}%)"

                            # ✅ 종료 커밋(조기커밋 못했을 때 최종 확정)
                            if cooldown == 0 and top1_p >= CONFIDENCE_THRESHOLD and margin >= MARGIN_TH:
                                if last_committed_kor != top1_kor:
                                    sentence_kor.append(top1_kor)
                                    sentence_eng.append(top1_eng)
                                    last_committed_kor = top1_kor

                                    last_commit_time = time.time()
                                    finalized_once = False
                                    final_eng_sent = ""
                                    final_kor_sent = ""

                                    print(f"[END COMMIT] {top1_label_num} {top1_kor} / {top1_eng}")

                                cooldown = COOLDOWN_FRAMES

                        segment_feats = []
                        ema_probs = None
                        early_stable_label = None
                        early_stable_cnt = 0
                        seg_frame_counter = 0

            # =========================
            # "Camera (Live only)" 창 UI (키보드 v2처럼)
            # =========================
            frame_ui = frame.copy()
            frame_ui = draw_panel(frame_ui, 15, 15, 900, 140, alpha=0.72)

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

            # =========================
            # Sentence 창 (키보드 v2 레이아웃 그대로)
            # =========================
            sent_canvas = np.zeros((420, 1200, 3), dtype=np.uint8)
            sent_canvas = draw_panel(sent_canvas, 0, 0, sent_canvas.shape[1], sent_canvas.shape[0], alpha=1.0)

            sent_canvas = put_korean_text(
                sent_canvas,
                "SENTENCE (CAM v2)",
                (20, 15),
                color=(200, 200, 200),
                font=_FONT_MAIN,
                stroke_width=3
            )

            # 상단: 커밋된 한국어 단어 나열
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

            gap_left = 0.0
            if last_commit_time is not None:
                gap_left = max(0.0, SENTENCE_GAP_SEC - (time.time() - last_commit_time)) if SENTENCE_GAP_SEC > 0 else 0.0

            sub = (
                f"ConfTH={CONFIDENCE_THRESHOLD:.2f} | Gap={SENTENCE_GAP_SEC:.1f}s | "
                f"TimeLeft={gap_left:.2f}s | Words={len(sentence_kor)}"
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

            final_show = final_kor_sent if final_kor_sent.strip() else "(아직 문장 확정 없음) - 0을 누르세요"
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

            # =========================
            # 키 입력
            # =========================
            key = cv2.waitKey(1) & 0xFF

            # 종료
            if key == ord("q"):
                break

            # REC 토글
            elif key == ord(" "):
                is_recording = not is_recording
                if is_recording:
                    feats_prev = None
                    motion_ema = 0.0
                    in_segment = False
                    end_hold_cnt = 0
                    segment_feats = []
                    ema_probs = None
                    cooldown = 0
                    latest_live_text = "CAM INPUT MODE"
                    early_stable_label = None
                    early_stable_cnt = 0
                    seg_frame_counter = 0

                    # 키보드 v2처럼: 새 REC 시작 시 문장도 초기화(원하면 주석)
                    sentence_kor.clear()
                    sentence_eng.clear()
                    last_committed_kor = None
                    last_commit_time = None
                    finalized_once = False
                    final_eng_sent = ""
                    final_kor_sent = ""
                    print("\n* REC ON")
                else:
                    latest_live_text = "-"
                    print("* REC OFF")

            # 리셋
            elif key == ord("r"):
                sentence_kor.clear()
                sentence_eng.clear()
                last_committed_kor = None
                last_commit_time = None
                finalized_once = False
                final_eng_sent = ""
                final_kor_sent = ""
                latest_live_text = "-"
                print("[RESET] cleared")

            # ✅ 0: 수동 Finalize (키보드 v2의 "0 + Enter"에 대응)
            elif key == ord("0"):
                if len(sentence_eng) > 0:
                    final_eng_sent, final_kor_sent = make_final_korean_sentence(
                        translator, sentence_eng, sentence_kor
                    )
                    finalized_once = True
                    print(f"[FINALIZE] EN={final_eng_sent} | KO={final_kor_sent}")
                else:
                    final_eng_sent, final_kor_sent = "", "(단어가 없어서 문장 확정 불가)"
                    finalized_once = True

    cap.release()
    cv2.destroyAllWindows()
    print("\n종료")
