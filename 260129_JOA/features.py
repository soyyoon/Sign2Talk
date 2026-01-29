import numpy as np

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

def calculate_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

def extract_hand_features(landmarks):
    features = []
    wrist = 0
    thumb_tip = 4
    index_tip = 8
    middle_tip = 12
    ring_tip = 16
    pinky_tip = 20

    for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
        v1 = np.array([landmarks[wrist].x - landmarks[tip].x,
                       landmarks[wrist].y - landmarks[tip].y])
        v2 = np.array([0, -1])
        features.append(float(calculate_angle(v1, v2)))

    tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    for i in range(len(tips) - 1):
        p1 = [landmarks[tips[i]].x, landmarks[tips[i]].y]
        p2 = [landmarks[tips[i+1]].x, landmarks[tips[i+1]].y]
        features.append(float(calculate_distance(p1, p2)))

    return features[:10]

def extract_hand_interaction(left_landmarks, right_landmarks):
    features = []
    key_points = [0, 4, 8, 12, 16, 20]

    for idx in key_points:
        p1 = [left_landmarks[idx].x, left_landmarks[idx].y]
        p2 = [right_landmarks[idx].x, right_landmarks[idx].y]
        features.append(float(calculate_distance(p1, p2)))

    left_wrist = [left_landmarks[0].x, left_landmarks[0].y]
    right_wrist = [right_landmarks[0].x, right_landmarks[0].y]
    features.append(float(calculate_distance(left_wrist, right_wrist)))

    for hand_lm in [left_landmarks, right_landmarks]:
        wrist = [hand_lm[0].x, hand_lm[0].y]
        middle = [hand_lm[9].x, hand_lm[9].y]
        direction = np.array([middle[0] - wrist[0], middle[1] - wrist[1]])
        for ref_dir in [[0, -1], [1, 0], [0, 1], [-1, 0]]:
            features.append(float(calculate_angle(direction, ref_dir)))

    return features[:15]

def extract_pose_features(landmarks):
    features = []
    for shoulder, elbow, wrist in [(11, 13, 15), (12, 14, 16)]:
        v1 = np.array([landmarks[elbow].x - landmarks[shoulder].x,
                       landmarks[elbow].y - landmarks[shoulder].y])
        v2 = np.array([landmarks[wrist].x - landmarks[elbow].x,
                       landmarks[wrist].y - landmarks[wrist].y])  # 원본 그대로? -> 아니, 원본은 elbow 기준이었음
        # ⚠️ 위 줄은 실수 위험. 원본대로 유지:
        # v2 = np.array([landmarks[wrist].x - landmarks[elbow].x,
        #                landmarks[wrist].y - landmarks[elbow].y])

        # 안전하게 원본 그대로로 덮어쓰기
        v2 = np.array([landmarks[wrist].x - landmarks[elbow].x,
                       landmarks[wrist].y - landmarks[elbow].y])

        features.append(float(calculate_angle(v1, v2)))

    shoulder_center_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_center_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_center_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_center_y = (landmarks[23].y + landmarks[24].y) / 2

    for wrist_idx in [15, 16]:
        features.append(float(landmarks[wrist_idx].x - shoulder_center_x))
        features.append(float(landmarks[wrist_idx].y - shoulder_center_y))
        features.append(float(landmarks[wrist_idx].x - hip_center_x))
        features.append(float(landmarks[wrist_idx].y - hip_center_y))

    shoulder_width = calculate_distance([landmarks[11].x, landmarks[11].y],
                                        [landmarks[12].x, landmarks[12].y])
    features.append(float(shoulder_width))

    torso_height = calculate_distance([shoulder_center_x, shoulder_center_y],
                                      [hip_center_x, hip_center_y])
    features.append(float(torso_height))

    return features[:12]

def extract_frame_features(results, feature_dim: int = 47):
    features = []

    if results.left_hand_landmarks:
        features.extend(extract_hand_features(results.left_hand_landmarks.landmark)[:10])
    else:
        features.extend([0.0] * 10)

    if results.right_hand_landmarks:
        features.extend(extract_hand_features(results.right_hand_landmarks.landmark)[:10])
    else:
        features.extend([0.0] * 10)

    if results.left_hand_landmarks and results.right_hand_landmarks:
        features.extend(extract_hand_interaction(
            results.left_hand_landmarks.landmark,
            results.right_hand_landmarks.landmark
        )[:15])
    else:
        features.extend([0.0] * 15)

    if results.pose_landmarks:
        features.extend(extract_pose_features(results.pose_landmarks.landmark)[:12])
    else:
        features.extend([0.0] * 12)

    if len(features) < feature_dim:
        features.extend([0.0] * (feature_dim - len(features)))
    elif len(features) > feature_dim:
        features = features[:feature_dim]

    return np.array(features, dtype=np.float32)

def motion_score(feats_now: np.ndarray, feats_prev: np.ndarray) -> float:
    if feats_prev is None:
        return 0.0
    return float(np.mean(np.abs(feats_now - feats_prev)))
