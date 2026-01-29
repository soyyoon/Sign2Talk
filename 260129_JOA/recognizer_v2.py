# sign2talk/recognizer_text_v2.py
import cv2
import numpy as np
import time

from .config import CAM_W, CAM_H, CONFIDENCE_THRESHOLD
from .labels import KOR_MAIN, ENG_MAIN
from .translator import EnKoTranslator, make_final_korean_sentence
from .ui import draw_panel, put_korean_text, wrap_text_by_chars, get_fonts


# =========================
# ✅ TEXT INPUT v2 설정
# =========================
SENTENCE_GAP_SEC = 1.5  # (자동 종료도 원하면 유지 가능) - 지금은 0키로 수동 종료 권장


def main():
    _FONT_MAIN, _FONT_SENT = get_fonts()

    # ✅ 번역 모델은 1번만 로드
    translator = EnKoTranslator()

    # =========================
    # 상태
    # =========================
    is_recording = False

    # 커밋된 단어들
    sentence_kor = []
    sentence_eng = []

    last_committed_kor = None
    last_commit_time = None

    # 문장 확정(번역 결과)
    finalized_once = False
    final_eng_sent = ""
    final_kor_sent = ""

    latest_live_text = "-"
    input_buffer = ""  # ✅ 숫자 입력 버퍼 (예: "71")

    print("=" * 70)
    print("✅ recognizer_text_v2.py (NO WEBCAM)")
    print(" - 카메라 없이 창 2개를 띄우고, 숫자 입력으로 라벨을 커밋합니다.")
    print(" - Space: Start/Stop (REC 토글)")
    print(" - Enter: 입력한 숫자(1~77) 커밋")
    print(" - 0: 문장 종료 -> 번역/검증 실행 (FINAL 출력)")
    print(" - Backspace: 입력 버퍼 지우기(한 글자 삭제)")
    print(" - r: reset")
    print(" - q: quit")
    print("=" * 70)

    # ✅ 창을 마우스로 마음대로 리사이즈 가능하게
    cv2.namedWindow("Camera (Live only)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sentence", cv2.WINDOW_NORMAL)

    while True:
        # =========================
        # "Camera (Live only)" 창: 검은 캔버스
        # =========================
        frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        frame = draw_panel(frame, 15, 15, 900, 140, alpha=0.72)

        if is_recording:
            cv2.circle(frame, (35, 55), 7, (0, 0, 255), -1)
            frame = put_korean_text(frame, "REC", (55, 35),
                                    color=(255, 255, 255), font=_FONT_MAIN, stroke_width=3)
        else:
            cv2.circle(frame, (35, 55), 7, (130, 130, 130), -1)
            frame = put_korean_text(frame, "IDLE", (55, 35),
                                    color=(220, 220, 220), font=_FONT_MAIN, stroke_width=3)

        live_show = latest_live_text if is_recording else "-"
        frame = put_korean_text(
            frame,
            f"LIVE: {live_show}",
            (25, 70),
            color=(0, 255, 255),
            font=_FONT_MAIN,
            stroke_width=4
        )

        # ✅ 입력 버퍼 표시
        frame = put_korean_text(
            frame,
            f"INPUT (1~77 then Enter) | 0=Finalize: {input_buffer}",
            (25, 105),
            color=(255, 255, 0),
            font=_FONT_MAIN,
            stroke_width=3
        )

        hint = "Space: REC | Enter: Commit | 0: Finalize | R: Reset | Q: Quit"
        frame = put_korean_text(
            frame,
            hint,
            (15, frame.shape[0] - 45),
            color=(255, 255, 255),
            font=_FONT_MAIN,
            stroke_width=3
        )

        cv2.imshow("Camera (Live only)", frame)

        # =========================
        # Sentence 창
        # =========================
        sent_canvas = np.zeros((420, 1200, 3), dtype=np.uint8)
        sent_canvas = draw_panel(sent_canvas, 0, 0, sent_canvas.shape[1], sent_canvas.shape[0], alpha=1.0)

        sent_canvas = put_korean_text(
            sent_canvas,
            "SENTENCE (TEXT INPUT v2)",
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
            gap_left = max(0.0, SENTENCE_GAP_SEC - (time.time() - last_commit_time))

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
        # (선택) 자동 문장 종료 감지(침묵) → 번역
        # - 지금은 "0키 finalize"가 주 방식이라, 자동은 원하면 켜두고 싫으면 주석처리
        # =========================
        if is_recording and (last_commit_time is not None) and (not finalized_once):
            if (time.time() - last_commit_time) >= SENTENCE_GAP_SEC and len(sentence_eng) > 0:
                final_eng_sent, final_kor_sent = make_final_korean_sentence(translator, sentence_eng, sentence_kor)
                finalized_once = True

        # =========================
        # 키 입력
        # =========================
        key = cv2.waitKey(1) & 0xFF

        # 종료
        if key == ord('q'):
            break

        # REC 토글
        elif key == ord(' '):
            is_recording = not is_recording
            if is_recording:
                latest_live_text = "TEXT INPUT MODE"
                print("\n* REC ON")
            else:
                latest_live_text = "-"
                print("* REC OFF")

        # 리셋
        elif key == ord('r'):
            sentence_kor.clear()
            sentence_eng.clear()
            last_committed_kor = None
            last_commit_time = None
            finalized_once = False
            final_eng_sent = ""
            final_kor_sent = ""
            input_buffer = ""
            latest_live_text = "-"
            print("[RESET] cleared")

        # =========================
        # ✅ REC ON 상태에서만 숫자 입력 처리
        # =========================
        if is_recording:
            # 숫자 입력: 0~9 모두 버퍼로
            if ord('0') <= key <= ord('9'):
                input_buffer += chr(key)

            # 백스페이스: 버퍼 한 글자 삭제
            elif key in (8, 127):
                input_buffer = input_buffer[:-1]

            # Enter: 버퍼 해석
            elif key in (13, 10):  # Enter
                buf = input_buffer.strip()

                # 1) 버퍼가 "0"이면: Finalize
                if buf == "0":
                    if len(sentence_eng) > 0:
                        final_eng_sent, final_kor_sent = make_final_korean_sentence(
                            translator, sentence_eng, sentence_kor
                        )
                        finalized_once = True
                        print(f"[FINALIZE] EN={final_eng_sent} | KO={final_kor_sent}")
                    else:
                        final_eng_sent, final_kor_sent = "", "(단어가 없어서 문장 확정 불가)"
                        finalized_once = True

                    input_buffer = ""  # finalize 후 버퍼 비움

                # 2) 그 외 숫자면: Commit (1~77)
                elif buf.isdigit():
                    label_num = int(buf)

                    if 1 <= label_num <= 77:
                        kor = KOR_MAIN.get(label_num, "UNKNOWN")
                        eng = ENG_MAIN.get(label_num, "unknown")

                        if last_committed_kor != kor:
                            sentence_kor.append(kor)
                            sentence_eng.append(eng)
                            last_committed_kor = kor

                            last_commit_time = time.time()
                            finalized_once = False
                            final_eng_sent = ""
                            final_kor_sent = ""

                            latest_live_text = f"{label_num} {kor} / {eng}"
                            print(f"[COMMIT] {label_num} {kor} / {eng}")
                    else:
                        print("[ERROR] label out of range (1~77):", label_num)

                    input_buffer = ""  # commit 후 버퍼 비움

                else:
                    # 숫자 아닌 이상한 입력
                    print("[ERROR] invalid input:", buf)
                    input_buffer = ""


    cv2.destroyAllWindows()
    print("\n종료")
