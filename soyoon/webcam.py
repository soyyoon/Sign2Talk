"""
실시간 수어 인식 (웹캠) - 한글 지원 버전
========================================
PIL을 사용하여 한글 폰트 렌더링

실행:
python webcam_local.py
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
from collections import deque
import time
from PIL import ImageFont, ImageDraw, Image

# ============================================================
# 설정
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MAX_FRAMES = 30
FEATURE_DIM = 53
CONFIDENCE_THRESHOLD = 0.85
FRAME_STRIDE = 10
DUPLICATE_FILTER_SEC = 2.0

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 한글 라벨 매핑
KSL_LABELS_KR = {
    "01": "안녕", "02": "뭐", "03": "만나다", "04": "비빔밥", "05": "반갑다",
    "06": "취미", "07": "나", "08": "영화", "09": "얼굴", "10": "보다",
    "11": "이름", "13": "감사하다", "14": "같다", "15": "미안하다",
    "16": "먹다", "17": "괜찮다", "18": "수고", "20": "나이",
    "21": "다시", "22": "몇", "23": "날", "24": "좋다", "25": "언제",
    "26": "우리", "27": "지하철", "29": "버스", "30": "타다",
    "31": "핸드폰", "32": "어디", "34": "위치",
    "36": "책임", "37": "누구", "38": "도착하다", "39": "가족", "40": "시간",
    "41": "소개", "42": "받다", "43": "묻다", "44": "걷다",
    "47": "여동생", "48": "공부하다", "49": "사람", "50": "지금",
    "51": "특별한", "52": "어제", "54": "시험", "55": "끝",
    "56": "너", "57": "걱정하다", "58": "결혼", "59": "노력", "60": "아니",
    "61": "땀", "62": "아직", "63": "마침내", "64": "태어나다", "65": "성공",
    "66": "부탁", "67": "서울", "68": "저녁", "69": "경험", "70": "초대",
    "71": "음식", "72": "원하다", "74": "한시간", "76": "잘", "77": "조심"
}


# ============================================================
# 한글 텍스트 그리기 (PIL 사용)
# ============================================================

def put_korean_text(img, text, pos, font_size=30, color=(255, 255, 255)):
    """
    PIL을 사용하여 한글 텍스트를 이미지에 그리기
    
    Args:
        img: OpenCV 이미지 (BGR)
        text: 표시할 텍스트
        pos: (x, y) 위치
        font_size: 폰트 크기
        color: (B, G, R) 색상
    """
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Mac 시스템 폰트 경로 (맥에 기본 설치된 한글 폰트)
    font_paths = [
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',  # AppleGothic
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',  # Apple SD Gothic Neo
        '/Library/Fonts/AppleGothic.ttf',
    ]
    
    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
    
    # 폰트를 찾지 못하면 기본 폰트 사용
    if font is None:
        font = ImageFont.load_default()
    
    # PIL은 RGB 색상 사용 (OpenCV는 BGR)
    color_rgb = (color[2], color[1], color[0])
    
    # 텍스트 그리기
    draw.text(pos, text, font=font, fill=color_rgb)
    
    # PIL 이미지를 다시 OpenCV 이미지로 변환
    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_result


# ============================================================
# Feature 추출 함수들 (이전과 동일)
# ============================================================

def calculate_distance(p1, p2):
    return float(np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2)))

def calculate_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    return float(np.arccos(cos_angle))

def get_53_features(results):
    feats = []
    if results.left_hand_landmarks:
        lm = results.left_hand_landmarks.landmark
        wrist, tips = 0, [4, 8, 12, 16, 20]
        for tip in tips:
            v1 = [lm[wrist].x - lm[tip].x, lm[wrist].y - lm[tip].y]
            feats.append(calculate_angle(v1, [0, -1]))
        for i in range(len(tips) - 1):
            feats.append(calculate_distance([lm[tips[i]].x, lm[tips[i]].y], [lm[tips[i + 1]].x, lm[tips[i + 1]].y]))
        feats.append(0.0)
    else:
        feats.extend([0.0] * 10)
    
    if results.right_hand_landmarks:
        lm = results.right_hand_landmarks.landmark
        wrist, tips = 0, [4, 8, 12, 16, 20]
        for tip in tips:
            v1 = [lm[wrist].x - lm[tip].x, lm[wrist].y - lm[tip].y]
            feats.append(calculate_angle(v1, [0, -1]))
        for i in range(len(tips) - 1):
            feats.append(calculate_distance([lm[tips[i]].x, lm[tips[i]].y], [lm[tips[i + 1]].x, lm[tips[i + 1]].y]))
        feats.append(0.0)
    else:
        feats.extend([0.0] * 10)
    
    if results.left_hand_landmarks and results.right_hand_landmarks:
        l_lm = results.left_hand_landmarks.landmark
        r_lm = results.right_hand_landmarks.landmark
        for i in [0, 4, 8, 12, 16, 20]:
            feats.append(calculate_distance([l_lm[i].x, l_lm[i].y], [r_lm[i].x, r_lm[i].y]))
        feats.append(calculate_distance([l_lm[0].x, l_lm[0].y], [r_lm[0].x, r_lm[0].y]))
        for hand in [l_lm, r_lm]:
            d = [hand[9].x - hand[0].x, hand[9].y - hand[0].y]
            for r in [[0, -1], [1, 0], [0, 1], [-1, 0]]:
                feats.append(calculate_angle(d, r))
    else:
        feats.extend([0.0] * 15)
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        for s, e, w in [(11, 13, 15), (12, 14, 16)]:
            v1 = [lm[e].x - lm[s].x, lm[e].y - lm[s].y]
            v2 = [lm[w].x - lm[e].x, lm[w].y - lm[e].y]
            feats.append(calculate_angle(v1, v2))
        sx, sy = (lm[11].x + lm[12].x) / 2, (lm[11].y + lm[12].y) / 2
        hx, hy = (lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2
        for w in [15, 16]:
            feats.extend([lm[w].x - sx, lm[w].y - sy, lm[w].x - hx, lm[w].y - hy])
        feats.append(calculate_distance([lm[11].x, lm[11].y], [lm[12].x, lm[12].y]))
        feats.append(calculate_distance([sx, sy], [hx, hy]))
    else:
        feats.extend([0.0] * 12)
    
    if results.face_landmarks:
        f_lm = results.face_landmarks.landmark
        if results.right_hand_landmarks:
            r_idx = results.right_hand_landmarks.landmark[8]
            for t in [1, 13, 263]:
                feats.append(calculate_distance([r_idx.x, r_idx.y], [f_lm[t].x, f_lm[t].y]))
        else:
            feats.extend([0.0] * 3)
        if results.left_hand_landmarks:
            l_idx = results.left_hand_landmarks.landmark[8]
            for t in [1, 13, 33]:
                feats.append(calculate_distance([l_idx.x, l_idx.y], [f_lm[t].x, f_lm[t].y]))
        else:
            feats.extend([0.0] * 3)
    else:
        feats.extend([0.0] * 6)
    
    if len(feats) < FEATURE_DIM:
        feats.extend([0.0] * (FEATURE_DIM - len(feats)))
    return np.array(feats[:FEATURE_DIM], dtype=np.float32)

def get_shoulder_width(results):
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        width = calculate_distance([lm[11].x, lm[11].y], [lm[12].x, lm[12].y])
        return max(width, 0.05)
    return 0.2

def spatial_normalization(features, shoulder_width):
    normalized = features.copy()
    distance_indices = list(range(5, 10)) + list(range(15, 20)) + list(range(20, 27)) + [45, 46] + list(range(47, 53))
    if shoulder_width > 0:
        normalized[distance_indices] /= shoulder_width
    return normalized


# ============================================================
# 수어 인식 클래스
# ============================================================

class SignLanguageRecognizer:
    def __init__(self, model_dir):
        print("\n" + "=" * 60)
        print("🤖 수어 인식 시스템 초기화")
        print("=" * 60)
        
        print("📦 모델 로딩 중...")
        self.models = []
        for i in range(1, 5):
            model_path = os.path.join(model_dir, f'model_v{i}.keras')
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path, compile=False)
                self.models.append(model)
                print(f"   ✓ V{i} 로드 완료")
        
        if len(self.models) == 0:
            raise FileNotFoundError(f"❌ 모델을 찾을 수 없습니다: {model_dir}")
        
        print("\n📊 정규화 파라미터 로딩...")
        self.mean = np.load(os.path.join(model_dir, 'mean.npy'))
        self.std = np.load(os.path.join(model_dir, 'std.npy'))
        
        print("\n⚖️ 앙상블 가중치 로딩...")
        weights_path = os.path.join(model_dir, 'ensemble_weights.npy')
        if os.path.exists(weights_path):
            self.ensemble_weights = np.load(weights_path)[:len(self.models)]
            self.ensemble_weights = self.ensemble_weights / self.ensemble_weights.sum()
        else:
            self.ensemble_weights = np.ones(len(self.models)) / len(self.models)
        print(f"   가중치: {[f'{w:.2f}' for w in self.ensemble_weights]}")
        
        print("\n📋 클래스 이름 로딩...")
        with open(os.path.join(model_dir, 'class_names.json'), 'r') as f:
            self.class_names = json.load(f)
        print(f"   {len(self.class_names)}개 클래스")
        
        self.frame_buffer = deque(maxlen=MAX_FRAMES)
        self.last_predictions = deque(maxlen=10)
        self.last_prediction_time = {}
        
        print("\n✅ 초기화 완료!")
        print("=" * 60 + "\n")
    
    def predict(self, features_sequence):
        normalized = (features_sequence - self.mean) / self.std
        X = np.expand_dims(normalized, axis=0)
        ensemble_probs = np.zeros(len(self.class_names))
        for model, weight in zip(self.models, self.ensemble_weights):
            if X.ndim == 4 and X.shape[1] == 1:
                X = np.squeeze(X, axis=1)
            probs = model.predict(X, verbose=0)[0]  
            ensemble_probs += probs * weight
        pred_idx = np.argmax(ensemble_probs)
        confidence = ensemble_probs[pred_idx]
        pred_class = self.class_names[pred_idx]
        return pred_class, confidence
    
    def add_frame(self, features, shoulder_width):
        normalized = spatial_normalization(features, shoulder_width)
        self.frame_buffer.append(normalized)
    
    def filter_duplicate(self, pred_class, confidence):
        if confidence < CONFIDENCE_THRESHOLD:
            return None
        current_time = time.time()
        if pred_class in self.last_prediction_time:
            if current_time - self.last_prediction_time[pred_class] < DUPLICATE_FILTER_SEC:
                return None
        self.last_prediction_time[pred_class] = current_time
        self.last_predictions.append((pred_class, confidence))
        return pred_class


# ============================================================
# UI (한글 지원)
# ============================================================

def draw_ui(frame, recognizer, pred_class, confidence, fps):
    """UI 그리기 - 한글 지원"""
    h, w = frame.shape[:2]
    
    # 반투명 상단 패널
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # 제목 (영어만)
    cv2.putText(frame, "Sign Language Recognition", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # 상태
    buffer_text = f"Buffer: {len(recognizer.frame_buffer)}/{MAX_FRAMES}"
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, f"{buffer_text}  |  {fps_text}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # 예측 결과 (한글 사용)
    if pred_class and confidence:
        color = (0, 255, 0) if confidence >= 0.9 else (0, 255, 255) if confidence >= 0.7 else (0, 165, 255)
        
        korean_label = KSL_LABELS_KR.get(pred_class, "")
        if korean_label:
            display_text = f"{pred_class}: {korean_label}"
        else:
            display_text = pred_class
        
        # 한글 텍스트 그리기
        frame = put_korean_text(frame, display_text, (20, 140), font_size=50, color=color)
        
        # 신뢰도 (영어)
        cv2.putText(frame, f"{confidence*100:.1f}%", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 최근 예측 (한글 사용)
    if recognizer.last_predictions:
        cv2.rectangle(overlay, (0, h-120), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        cv2.putText(frame, "Recent:", (20, h-90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        recent = list(recognizer.last_predictions)[-5:]
        text_parts = []
        for w, c in recent:
            kr = KSL_LABELS_KR.get(w, "")
            if kr:
                text_parts.append(f"{w}:{kr}({c*100:.0f}%)")
            else:
                text_parts.append(f"{w}({c*100:.0f}%)")
        text = " > ".join(text_parts)
        
        # 한글 히스토리
        frame = put_korean_text(frame, text, (20, h-60), font_size=18, color=(200, 200, 200))
    
    # 도움말
    cv2.putText(frame, "Press 'Q' to quit | 'R' to reset", (20, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return frame


# ============================================================
# 메인
# ============================================================

def main():
    recognizer = SignLanguageRecognizer(MODEL_DIR)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("🎥 웹캠 시작!")
    print("=" * 60)
    print("💡 팁:")
    print("   - 손을 화면에 잘 보이게 하세요")
    print("   - Buffer가 30/30이 될 때까지 기다리세요")
    print("   - 천천히, 정확하게 수어를 수행하세요")
    print("=" * 60 + "\n")
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    current_prediction = None
    current_confidence = None
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = holistic.process(frame_rgb)
            
            features = get_53_features(results)
            shoulder_width = get_shoulder_width(results)
            recognizer.add_frame(features, shoulder_width)
            
            if len(recognizer.frame_buffer) == MAX_FRAMES and frame_count % FRAME_STRIDE == 0:
                features_seq = np.array(list(recognizer.frame_buffer))
                pred_class, confidence = recognizer.predict(features_seq)
                
                filtered = recognizer.filter_duplicate(pred_class, confidence)
                if filtered:
                    current_prediction = pred_class
                    current_confidence = confidence
                    korean = KSL_LABELS_KR.get(pred_class, "")
                    if korean:
                        print(f"✅ {pred_class}: {korean} ({confidence*100:.1f}%)")
                    else:
                        print(f"✅ {pred_class} ({confidence*100:.1f}%)")
            
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_time)
                fps_time = time.time()
            
            frame = draw_ui(frame, recognizer, current_prediction, current_confidence, fps)
            
            cv2.imshow('Sign Language Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                recognizer.frame_buffer.clear()
                recognizer.last_predictions.clear()
                current_prediction = None
                print("🔄 리셋")
    
    except KeyboardInterrupt:
        print("\n⏹️ 종료 중...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("✅ 종료 완료")


if __name__ == "__main__":
    main()