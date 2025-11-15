import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# 커스텀 손실 함수 정의 (모델 학습 시 사용한 것)
def smooth_sparse_categorical_crossentropy(y_true, y_pred, label_smoothing=0.1):
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

    y_true_smoothed = y_true_one_hot * (1 - label_smoothing) + (
        label_smoothing / tf.cast(num_classes, tf.float32)
    )

    loss = -tf.reduce_sum(
        y_true_smoothed * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)), axis=-1
    )
    return tf.reduce_mean(loss)


# 모델 로드
MODEL_PATH = "DIR_TO_MODEL/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# 클래스 이름 (67개 클래스)
CLASS_NAMES = [f"{i:02d}" for i in range(1, 68)]  # 01부터 67까지

# 설정
TARGET_FRAMES = 30
FEATURE_DIM = 47
CONFIDENCE_THRESHOLD = 0.7


def calculate_angle(v1, v2):
    """두 벡터 사이의 각도 계산"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0

    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def calculate_distance(p1, p2):
    """유클리드 거리"""
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def extract_hand_features(landmarks):
    """손 특징 (정확히 10차원)"""
    features = []

    wrist = 0
    thumb_tip = 4
    index_tip = 8
    middle_tip = 12
    ring_tip = 16
    pinky_tip = 20

    # 손가락 각도 (5차원)
    for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
        v1 = np.array(
            [
                landmarks[wrist].x - landmarks[tip].x,
                landmarks[wrist].y - landmarks[tip].y,
            ]
        )
        v2 = np.array([0, -1])
        angle = calculate_angle(v1, v2)
        features.append(float(angle))

    # 손가락 간 거리 (5차원)
    tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    for i in range(len(tips) - 1):
        p1 = [landmarks[tips[i]].x, landmarks[tips[i]].y]
        p2 = [landmarks[tips[i + 1]].x, landmarks[tips[i + 1]].y]
        dist = calculate_distance(p1, p2)
        features.append(float(dist))

    return features[:10]


def extract_hand_interaction(left_landmarks, right_landmarks):
    """양손 상호작용 (정확히 15차원)"""
    features = []

    key_points = [0, 4, 8, 12, 16, 20]

    # 양손 간 거리 (6차원)
    for idx in key_points:
        p1 = [left_landmarks[idx].x, left_landmarks[idx].y]
        p2 = [right_landmarks[idx].x, right_landmarks[idx].y]
        dist = calculate_distance(p1, p2)
        features.append(float(dist))

    # 손목 거리 (1차원)
    left_wrist = [left_landmarks[0].x, left_landmarks[0].y]
    right_wrist = [right_landmarks[0].x, right_landmarks[0].y]
    wrist_dist = calculate_distance(left_wrist, right_wrist)
    features.append(float(wrist_dist))

    # 손 방향 (8차원)
    for hand_lm in [left_landmarks, right_landmarks]:
        wrist = [hand_lm[0].x, hand_lm[0].y]
        middle = [hand_lm[9].x, hand_lm[9].y]

        direction = np.array([middle[0] - wrist[0], middle[1] - wrist[1]])

        for ref_dir in [[0, -1], [1, 0], [0, 1], [-1, 0]]:
            angle = calculate_angle(direction, ref_dir)
            features.append(float(angle))

    return features[:15]


def extract_pose_features(landmarks):
    """포즈 특징 (정확히 12차원)"""
    features = []

    # 팔 각도 (2차원)
    for shoulder, elbow, wrist in [(11, 13, 15), (12, 14, 16)]:
        v1 = np.array(
            [
                landmarks[elbow].x - landmarks[shoulder].x,
                landmarks[elbow].y - landmarks[shoulder].y,
            ]
        )
        v2 = np.array(
            [
                landmarks[wrist].x - landmarks[elbow].x,
                landmarks[wrist].y - landmarks[elbow].y,
            ]
        )
        angle = calculate_angle(v1, v2)
        features.append(float(angle))

    # 몸통 대비 팔 위치 (8차원)
    shoulder_center_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_center_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_center_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_center_y = (landmarks[23].y + landmarks[24].y) / 2

    for wrist_idx in [15, 16]:
        features.append(float(landmarks[wrist_idx].x - shoulder_center_x))
        features.append(float(landmarks[wrist_idx].y - shoulder_center_y))
        features.append(float(landmarks[wrist_idx].x - hip_center_x))
        features.append(float(landmarks[wrist_idx].y - hip_center_y))

    # 어깨 너비 (1차원)
    shoulder_width = calculate_distance(
        [landmarks[11].x, landmarks[11].y], [landmarks[12].x, landmarks[12].y]
    )
    features.append(float(shoulder_width))

    # 몸통 높이 (1차원)
    torso_height = calculate_distance(
        [shoulder_center_x, shoulder_center_y], [hip_center_x, hip_center_y]
    )
    features.append(float(torso_height))

    return features[:12]


def extract_frame_features(results):
    """47차원 특징 추출"""
    features = []

    # 왼손 (10차원)
    if results.left_hand_landmarks:
        left_features = extract_hand_features(results.left_hand_landmarks.landmark)
        features.extend(left_features[:10])
    else:
        features.extend([0] * 10)

    # 오른손 (10차원)
    if results.right_hand_landmarks:
        right_features = extract_hand_features(results.right_hand_landmarks.landmark)
        features.extend(right_features[:10])
    else:
        features.extend([0] * 10)

    # 양손 상호작용 (15차원)
    if results.left_hand_landmarks and results.right_hand_landmarks:
        interaction_features = extract_hand_interaction(
            results.left_hand_landmarks.landmark, results.right_hand_landmarks.landmark
        )
        features.extend(interaction_features[:15])
    else:
        features.extend([0] * 15)

    # 포즈 (12차원)
    if results.pose_landmarks:
        pose_features = extract_pose_features(results.pose_landmarks.landmark)
        features.extend(pose_features[:12])
    else:
        features.extend([0] * 12)

    # 정확히 47개 보장
    if len(features) < 47:
        features.extend([0] * (47 - len(features)))
    elif len(features) > 47:
        features = features[:47]

    return np.array(features, dtype=np.float32)


def predict_sign(frame_buffer):
    """수어 예측"""
    if len(frame_buffer) < TARGET_FRAMES:
        return None, 0.0

    # (30, 47) 형태로 변환
    features = np.array(list(frame_buffer)[-TARGET_FRAMES:])

    # 정규화 (학습 시 사용한 것과 동일)
    mean = features.mean(axis=(0, 1), keepdims=True)
    std = features.std(axis=(0, 1), keepdims=True) + 1e-6
    features_normalized = (features - mean) / std

    # 배치 차원 추가 (1, 30, 47)
    features_batch = np.expand_dims(features_normalized, axis=0)

    # 예측
    predictions = model.predict(features_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    return CLASS_NAMES[predicted_class], confidence


def main():
    """메인 실행 함수"""
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다!")
        return

    # 프레임 버퍼 (최근 30프레임 저장)
    frame_buffer = deque(maxlen=TARGET_FRAMES)

    # 상태 변수
    is_recording = False
    predicted_sign = ""
    confidence = 0.0
    frame_count = 0

    print("=" * 60)
    print("수어 인식 시작")
    print("=" * 60)
    print("사용법:")
    print("  - 스페이스바: 녹화 시작/중지")
    print("  - 'q': 종료")
    print("=" * 60)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("!!! 프레임을 읽을 수 없음")
                break

            # 좌우 반전
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe 처리
            results = holistic.process(frame_rgb)

            # 랜드마크 그리기
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
                )
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )

            # 녹화 중일 때
            if is_recording:
                # 특징 추출
                features = extract_frame_features(results)
                frame_buffer.append(features)
                frame_count += 1

                # 30프레임 모이면 예측
                if len(frame_buffer) == TARGET_FRAMES:
                    predicted_sign, confidence = predict_sign(frame_buffer)
                    print(f"예측 완료: {predicted_sign} ({confidence*100:.1f}%)")

                    # 자동으로 녹화 중지
                    is_recording = False
                    frame_count = 0
                    frame_buffer.clear()

            # UI 그리기
            # 상태 표시
            status_text = "녹화 중..." if is_recording else "대기 중 (스페이스바)"
            status_color = (0, 0, 255) if is_recording else (255, 255, 255)
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                status_color,
                2,
            )

            # 프레임 카운트
            if is_recording:
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}/{TARGET_FRAMES}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # 예측 결과 표시
            if predicted_sign and confidence > CONFIDENCE_THRESHOLD:
                result_text = f"Sign: {predicted_sign} ({confidence*100:.1f}%)"
                cv2.putText(
                    frame,
                    result_text,
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                )
            elif predicted_sign:
                result_text = (
                    f"Sign: {predicted_sign} ({confidence*100:.1f}%) - Low confidence"
                )
                cv2.putText(
                    frame,
                    result_text,
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                )

            # 화면 표시
            cv2.imshow("Real-time Sign Language Recognition", frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):  # 스페이스바
                if not is_recording:
                    is_recording = True
                    frame_buffer.clear()
                    frame_count = 0
                    predicted_sign = ""
                    confidence = 0.0
                    print("\n* 녹화 시작...")
                else:
                    is_recording = False
                    frame_count = 0
                    print("* 녹화 중지")

    cap.release()
    cv2.destroyAllWindows()
    print("\n종료")


if __name__ == "__main__":
    main()
