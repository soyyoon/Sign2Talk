"""
실시간 수어 인식 - 연속 인식 + 자동 문장 생성
========================================
Space 1번: 녹화 시작 (실시간 연속 인식)
Space 2번: 녹화 종료 + 문장 생성
R: 리셋
Q: 종료
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
import openai
from openai import OpenAI
from translator import Translator, make_final_korean_sentence

# ============================================================
# 설정
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# OpenAI API 설정 (환경변수에서 가져오기)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print("⚠️  경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다!")
    OPENAI_API_KEY = "your-api-key-here"

# OpenAI 클라이언트 초기화
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    USE_NEW_API = True
except:
    openai.api_key = OPENAI_API_KEY
    USE_NEW_API = False

MAX_FRAMES = 30
FEATURE_DIM = 53
CONFIDENCE_THRESHOLD = 0.70  # 실시간 인식이니 조금 낮게
FRAME_STRIDE = 10  

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
# 한글 텍스트 그리기
# ============================================================

def put_korean_text(img, text, pos, font_size=30, color=(255, 255, 255)):
    """PIL을 사용하여 한글 텍스트 그리기"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_paths = [
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
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
    
    if font is None:
        font = ImageFont.load_default()
    
    color_rgb = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=color_rgb)
    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img_result


# ============================================================
# Feature 추출 (동일)
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
        
        # 녹화 상태
        self.is_recording = False
        self.frame_buffer = deque(maxlen=MAX_FRAMES)
        
        # 인식된 단어 리스트
        self.word_list = []
        
        # 최근 인식 정보 (중복 방지)
        self.last_predicted_word = None
        self.last_prediction_time = 0
        
        # 실시간 예측 표시
        self.current_prediction = None
        self.current_confidence = 0.0
        
        # 생성된 문장
        self.generated_sentence = ""
        self.generating = False
        
        # 문장 생성기 (GPT)
        self.translator = Translator()

        print("\n✅ 초기화 완료!")
        print("=" * 60 + "\n")

    def push_word(self, pred_class, current_time):
        """같은 단어가 연속으로 들어가지 않도록 추가"""
        if self.word_list and self.word_list[-1] == pred_class:
            self.last_predicted_word = pred_class
            self.last_prediction_time = current_time
            return False

        self.word_list.append(pred_class)
        self.last_predicted_word = pred_class
        self.last_prediction_time = current_time
        return True

    def pop_last_word(self):
        """백스페이스: 최근 단어 1개 삭제"""
        if not self.word_list:
            return None

        removed = self.word_list.pop()

        # 단어 리스트가 바뀌면 기존 생성 문장은 무효 -> 지움
        self.generated_sentence = ""

        # 삭제 후 최근 단어 갱신
        if self.word_list:
            self.last_predicted_word = self.word_list[-1]
            self.last_prediction_time = time.time()
        else:
            self.last_predicted_word = None
            self.last_prediction_time = 0

        return removed
        
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
    
    def add_frame_and_predict(self, features, shoulder_width, frame_count):
        """
        프레임 추가 + 실시간 예측
        녹화 중일 때만 작동
        """
        if not self.is_recording:
            return
        
        # 프레임 추가
        normalized = spatial_normalization(features, shoulder_width)
        self.frame_buffer.append(normalized)
        
        # 버퍼가 차고, stride마다 예측
        if len(self.frame_buffer) == MAX_FRAMES and frame_count % FRAME_STRIDE == 0:
            features_seq = np.array(list(self.frame_buffer))
            pred_class, confidence = self.predict(features_seq)
            
            # 실시간 표시 업데이트
            self.current_prediction = pred_class
            self.current_confidence = confidence
            
            # 신뢰도 체크 + 중복 체크
            current_time = time.time()
            is_new_word = (
                confidence >= CONFIDENCE_THRESHOLD and
                (pred_class != self.last_predicted_word or 
                 current_time - self.last_prediction_time > 2.5)  # 1.5초 간격
            )
            
            if is_new_word:
                self.word_list.append(pred_class)
                self.last_predicted_word = pred_class
                self.last_prediction_time = current_time
                
                korean = KSL_LABELS_KR.get(pred_class, pred_class)
                print(f"✅ {pred_class}: {korean} ({confidence*100:.1f}%)")
    
    def start_recording(self):
        """녹화 시작"""
        self.is_recording = True
        self.frame_buffer.clear()
        self.word_list.clear()
        self.last_predicted_word = None
        self.generated_sentence = ""
        self.current_prediction = None
        print("🔴 녹화 시작 - 실시간 인식 활성화")
    
    def stop_recording_and_generate(self):
        """녹화 종료 + 즉시 문장 생성"""
        self.is_recording = False
        self.current_prediction = None
        
        print(f"⏹️  녹화 종료")
        print(f"   인식된 단어: {len(self.word_list)}개")
        
        if not self.word_list:
            self.generated_sentence = "인식된 단어가 없습니다."
            print("   ⚠️  단어 없음")
            return
        
        # GPT 문장 생성
        self.generating = True
        print("💬 GPT로 문장 생성 중...")
        _, ko_sentence = make_final_korean_sentence(
            translator=self.translator,
            sentence_eng=[],                 # 영어는 디버그용 → 비워도 됨
            sentence_kor=[KSL_LABELS_KR.get(w, w) for w in self.word_list]
        )
        self.generated_sentence = ko_sentence
        self.generating = False
        print(f"✅ 생성 완료: {self.generated_sentence}\n")
    
    def reset(self):
        """리셋"""
        self.is_recording = False
        self.frame_buffer.clear()
        self.word_list.clear()
        self.generated_sentence = ""
        self.current_prediction = None
        self.last_predicted_word = None
        print("🔄 전체 리셋")


# ============================================================
# UI
# ============================================================

def draw_ui(frame, recognizer, fps):
    """UI 그리기"""
    h, w = frame.shape[:2]
    
    # 상단 패널
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # 제목
    cv2.putText(frame, "Sign Language Recognition", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # 상태
    if recognizer.generating:
        status_text = "GENERATING..."
        status_color = (255, 200, 0)
    elif recognizer.is_recording:
        status_text = "RECORDING"
        status_color = (0, 0, 255)
    else:
        status_text = "READY"
        status_color = (0, 255, 0)
    
    cv2.putText(frame, status_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # 버퍼 상태 + FPS
    buffer_text = f"Buffer: {len(recognizer.frame_buffer)}/{MAX_FRAMES}"
    cv2.putText(frame, f"{buffer_text}  |  FPS: {fps:.1f}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # 실시간 예측 표시 (오른쪽 상단)
    if recognizer.is_recording and recognizer.current_prediction:
        korean = KSL_LABELS_KR.get(recognizer.current_prediction, "")
        if korean:
            pred_text = f"{recognizer.current_prediction}:{korean}"
        else:
            pred_text = recognizer.current_prediction
        
        conf = recognizer.current_confidence
        color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (100, 100, 100)
        
        frame = put_korean_text(frame, pred_text, (w - 250, 40), 
                                font_size=25, color=color)
        cv2.putText(frame, f"{conf*100:.0f}%", (w - 100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 인식된 단어 리스트 (화면 하단에 가로로 나열)
    if recognizer.word_list:
        # 하단 배경
        cv2.rectangle(overlay, (0, h-180), (w, h-80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        cv2.putText(frame, "Words:", (20, h-150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 단어들을 가로로 나열
        words_display = []
        for word in recognizer.word_list:
            korean = KSL_LABELS_KR.get(word, word)
            words_display.append(f"{word}:{korean}")
        
        # 한 줄로 표시
        words_text = "  →  ".join(words_display)
        frame = put_korean_text(frame, words_text, (20, h-115), 
                                font_size=22, color=(100, 255, 255))
    
    # 생성된 문장 (맨 아래)
    if recognizer.generated_sentence:
        cv2.rectangle(overlay, (0, h-70), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        
        frame = put_korean_text(frame, f"문장: {recognizer.generated_sentence}", 
                                (20, h-45), font_size=23, color=(100, 255, 100))
    
    # 도움말
    help_text = "Space: Start/Stop & Generate | R: Reset | Q: Quit"
    cv2.putText(frame, help_text, (20, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return frame


# ============================================================
# 메인
# ============================================================

def main():
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
        print("⚠️  경고: OpenAI API 키가 설정되지 않았습니다!")
        print("   export OPENAI_API_KEY='sk-proj-xxxxx...'\n")
    
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
    print("💡 사용법:")
    print("   Space 1번: 녹화 시작 (실시간 연속 인식)")
    print("   Space 2번: 녹화 종료 + 문장 생성")
    print("   R: 리셋")
    print("   Q: 종료")
    print("=" * 60 + "\n")
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = holistic.process(frame_rgb)
            
            # Feature 추출 + 실시간 예측
            features = get_53_features(results)
            shoulder_width = get_shoulder_width(results)
            recognizer.add_frame_and_predict(features, shoulder_width, frame_count)
            
            # 랜드마크 그리기
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_time)
                fps_time = time.time()
            
            # UI 그리기
            frame = draw_ui(frame, recognizer, fps)
            
            cv2.imshow('Sign Language Recognition', frame)
            
            # 키보드 입력
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            
            elif key == ord(' '):  # Space
                if not recognizer.is_recording:
                    recognizer.start_recording()
                else:
                    recognizer.stop_recording_and_generate()
            
            elif key == ord('r') or key == ord('R'):
                recognizer.reset()
    
    except KeyboardInterrupt:
        print("\n⏹️ 종료 중...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("✅ 종료 완료")


if __name__ == "__main__":
    main()
