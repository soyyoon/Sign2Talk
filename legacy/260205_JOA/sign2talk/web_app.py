# sign2talk/web_app.py
import queue
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from . import recognizer

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

_latest_frame_lock = threading.Lock()
_latest_frame = None
_last_frame_time = 0.0

_key_queue = queue.Queue()

# 웹 스트리밍 최적화 (렉 줄이기)
WEB_STREAM_WIDTH = 720
WEB_STREAM_FPS = 10            # ✅ 15 -> 10 (인코딩 병목이면 큰 차이)
WEB_JPEG_QUALITY = 50          # ✅ 60 -> 50


def _frame_callback(frame: np.ndarray):
    """recognizer.main 에서 호출되는 콜백: 최신 프레임을 저장"""
    global _latest_frame, _last_frame_time
    now = time.time()

    # 프레임 레이트 제한
    if now - _last_frame_time < 1.0 / WEB_STREAM_FPS:
        return
    _last_frame_time = now

    # 해상도 리사이즈
    h, w = frame.shape[:2]
    if w > WEB_STREAM_WIDTH:
        scale = WEB_STREAM_WIDTH / w
        new_w = WEB_STREAM_WIDTH
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # ✅ 속도 우선
    else:
        frame_resized = frame

    with _latest_frame_lock:
        _latest_frame = frame_resized.copy()


def _get_latest_frame():
    with _latest_frame_lock:
        if _latest_frame is None:
            return None
        return _latest_frame.copy()


def _start_recognizer_thread():
    t = threading.Thread(
        target=recognizer.main,
        kwargs={
            "frame_callback": _frame_callback,
            "use_windows": False,
            "key_queue": _key_queue,
            "draw_landmarks": False,  # 웹에서는 랜드마크 draw 끄기(렉 방지)
        },
        daemon=True,
    )
    t.start()


@app.route("/")
def index():
    return render_template("index.html")


def _mjpeg_generator():
    """브라우저로 MJPEG 스트림 전송"""
    frame_time = 1.0 / WEB_STREAM_FPS
    last_sent_time = 0.0

    while True:
        now = time.time()
        if now - last_sent_time < frame_time:
            time.sleep(0.02)
            continue

        frame = _get_latest_frame()
        if frame is None:
            time.sleep(frame_time)
            continue

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, WEB_JPEG_QUALITY]
        ok, encoded = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            time.sleep(frame_time)
            continue

        jpg_bytes = encoded.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
        )

        last_sent_time = time.time()


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/key", methods=["POST"])
def handle_key():
    data = request.get_json()
    key_char = data.get("key", "")
    if key_char:
        _key_queue.put(ord(key_char))
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"}), 400


def run():
    _start_recognizer_thread()
    app.run(debug=False)


if __name__ == "__main__":
    run()
