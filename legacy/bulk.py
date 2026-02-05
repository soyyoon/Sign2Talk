import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

mp_holistic = mp.solutions.holistic


def calculate_angle(v1, v2):
    # 두 벡터 사이의 각도 계산 (rad)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0

    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def calculate_distance(p1, p2):
    # 두 점 사이의 유클리드 거리 계산
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def project_point_to_plane(point, plane_origin, plane_normal):
    # 점을 평면에 투영
    v = np.array(point) - np.array(plane_origin)
    distance = np.dot(v, plane_normal)
    projected = np.array(point) - distance * plane_normal
    return projected


def extract_feature_data(results, frame_width, frame_height):
    """
    47개의 feature data 추출
    Returns: numpy array of shape (47,)
    """
    features = []

    # landmarks가 없으면 0으로
    if (
        not results.pose_landmarks
        or not results.left_hand_landmarks
        or not results.right_hand_landmarks
    ):
        return np.zeros(47)

    # 좌표 변환
    def get_coords(landmark):
        return np.array([landmark.x, landmark.y, landmark.z])

    # Pose landmarks
    pose = results.pose_landmarks.landmark
    left_hand = results.left_hand_landmarks.landmark
    right_hand = results.right_hand_landmarks.landmark
    face = results.face_landmarks.landmark if results.face_landmarks else None

    # 주요 포인트 미리 계산
    left_shoulder = get_coords(pose[11])
    right_shoulder = get_coords(pose[12])
    shoulder_center = (left_shoulder + right_shoulder) / 2
    left_elbow = get_coords(pose[13])
    right_elbow = get_coords(pose[14])
    left_wrist = get_coords(pose[15])
    right_wrist = get_coords(pose[16])
    nose = get_coords(pose[0])

    # 1. 손가락 사이 각도 (8개)
    for hand_landmarks in [left_hand, right_hand]:
        mcp_indices = [5, 9, 13, 17]
        tip_indices = [8, 12, 16, 20]
        for i in range(4):
            mcp = get_coords(hand_landmarks[mcp_indices[i]])
            current_tip = get_coords(hand_landmarks[tip_indices[i]])
            next_tip = (
                get_coords(hand_landmarks[tip_indices[i + 1]])
                if i < 3
                else get_coords(hand_landmarks[4])
            )
            v1 = current_tip - mcp
            v2 = next_tip - mcp
            features.append(calculate_angle(v1, v2))

    # 2. 손가락 구부림 각도 (10개)
    for hand_landmarks in [left_hand, right_hand]:
        finger_pip_indices = [2, 6, 10, 14, 18]
        finger_dip_indices = [3, 7, 11, 15, 19]
        finger_mcp_indices = [1, 5, 9, 13, 17]
        for i in range(5):
            pip = get_coords(hand_landmarks[finger_pip_indices[i]])
            dip = get_coords(hand_landmarks[finger_dip_indices[i]])
            mcp = get_coords(hand_landmarks[finger_mcp_indices[i]])
            v1 = dip - pip
            v2 = mcp - pip
            features.append(calculate_angle(v1, v2))

    # 3. 손가락 길이 (10개)
    for hand_landmarks in [left_hand, right_hand]:
        wrist = get_coords(hand_landmarks[0])
        tip_indices = [4, 8, 12, 16, 20]
        distances = []
        for tip_idx in tip_indices:
            tip = get_coords(hand_landmarks[tip_idx])
            dist = calculate_distance(wrist, tip)
            distances.append(dist)
        max_dist = max(distances) if max(distances) > 0 else 1
        for dist in distances:
            features.append(dist / max_dist)

    # 4. 손목 각도 (2개)
    for wrist, elbow, hand_landmarks in [
        (left_wrist, left_elbow, left_hand),
        (right_wrist, right_elbow, right_hand),
    ]:
        middle_mcp = get_coords(hand_landmarks[9])
        v1 = middle_mcp - wrist
        v2 = elbow - wrist
        features.append(calculate_angle(v1, v2))

    # 5. 팔꿈치 각도 (2개)
    for elbow, wrist, shoulder in [
        (left_elbow, left_wrist, left_shoulder),
        (right_elbow, right_wrist, right_shoulder),
    ]:
        v1 = wrist - elbow
        v2 = shoulder - elbow
        features.append(calculate_angle(v1, v2))

    # 6. 신체 길이 비율 계산
    d_body = calculate_distance(left_shoulder, right_shoulder)
    d_body = max(d_body, 1e-6)

    # 6-1. 팔 길이 (2개)
    for wrist, elbow in [(left_wrist, left_elbow), (right_wrist, right_elbow)]:
        dist = calculate_distance(wrist, elbow)
        features.append(dist / d_body)

    # 6-2. 손목-어깨 (2개)
    for wrist, shoulder in [(left_wrist, left_shoulder), (right_wrist, right_shoulder)]:
        dist = calculate_distance(wrist, shoulder)
        features.append(dist / d_body)

    # 6-3. 양손 사이 거리 (1개)
    hand_dist = calculate_distance(left_wrist, right_wrist)
    features.append(hand_dist / d_body)

    # 6-4. 팔뚝 길이 (2개)
    for elbow, shoulder in [(left_elbow, left_shoulder), (right_elbow, right_shoulder)]:
        dist = calculate_distance(elbow, shoulder)
        features.append(dist / d_body)

    # 6-5. 어깨 길이 (1개)
    features.append(d_body / d_body)

    # 7. 얼굴/손 관련
    if face:
        upper_lip_top = get_coords(face[13])
        lower_lip_bottom = get_coords(face[14])
        left_lip = get_coords(face[61])
        right_lip = get_coords(face[291])

        d_ul = calculate_distance(left_lip, right_lip)
        d_ul = max(d_ul, 1e-6)

        # 입 벌림 (1개)
        d_ol = calculate_distance(upper_lip_top, lower_lip_bottom)
        features.append(d_ol / d_ul)

        # 입 돌출 (1개)
        lip_center_lr = (left_lip + right_lip) / 2
        lip_center_tb = (upper_lip_top + lower_lip_bottom) / 2
        d_op = calculate_distance(lip_center_lr, lip_center_tb)
        features.append(d_op / d_ul)
    else:
        features.extend([0, 0])

    # 7-2. 양손과 얼굴 거리 (2개)
    d_wn_left = calculate_distance(left_wrist, nose)
    d_wn_right = calculate_distance(right_wrist, nose)
    features.append(d_wn_left / d_body)
    features.append(d_wn_right / d_body)

    # 8. 머리 자세 (3개)
    v_side = right_shoulder - left_shoulder
    v_side = v_side / (np.linalg.norm(v_side) + 1e-6)
    v_front_assumed = np.array([0, 0, -1])
    v_top = np.cross(v_side, v_front_assumed)
    v_top = v_top / (np.linalg.norm(v_top) + 1e-6)
    v_front = np.cross(v_top, v_side)
    v_front = v_front / (np.linalg.norm(v_front) + 1e-6)

    left_eye = get_coords(pose[2])
    right_eye = get_coords(pose[5])
    eye_center = (left_eye + right_eye) / 2

    # 8-1. 고개 좌우 돌림 (1개)
    proj_left_eye = project_point_to_plane(left_eye, shoulder_center, v_top)
    proj_right_eye = project_point_to_plane(right_eye, shoulder_center, v_top)
    v_eyes_proj = proj_right_eye - proj_left_eye
    v_eyes_proj = v_eyes_proj / (np.linalg.norm(v_eyes_proj) + 1e-6)
    yaw_angle = calculate_angle(v_eyes_proj, v_side)
    if np.dot(v_eyes_proj, v_front) > 0:
        yaw_angle = -yaw_angle
    features.append(yaw_angle)

    # 8-2. 고개 앞뒤 젖힘 (1개)
    proj_eye_center = project_point_to_plane(eye_center, shoulder_center, v_side)
    v_head = proj_eye_center - shoulder_center
    v_head = v_head / (np.linalg.norm(v_head) + 1e-6)
    pitch_angle = calculate_angle(v_head, v_top)
    if np.dot(v_head, v_front) < 0:
        pitch_angle = -pitch_angle
    features.append(pitch_angle)

    # 8-3. 고개 좌우 기울기 (1개)
    proj_left_eye_front = project_point_to_plane(left_eye, shoulder_center, v_front)
    proj_right_eye_front = project_point_to_plane(right_eye, shoulder_center, v_front)
    v_eyes_tilt = proj_right_eye_front - proj_left_eye_front
    v_eyes_tilt = v_eyes_tilt / (np.linalg.norm(v_eyes_tilt) + 1e-6)
    roll_angle = calculate_angle(v_eyes_tilt, v_side)
    if np.dot(v_eyes_tilt, v_top) < 0:
        roll_angle = -roll_angle
    features.append(roll_angle)

    return np.array(features)


def extract_landmarks(video_path):
    # 비디오에서 47개의 feature data 추출

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_features = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,  # 2->1, 속도향상
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # MediaPipe
            results = holistic.process(image)

            # Feature extraction
            features = extract_feature_data(results, frame_width, frame_height)
            all_features.append(features)

    cap.release()

    if len(all_features) == 0:
        return None

    # numpy 변환
    all_features = np.array(all_features)

    if all_features.ndim != 2:
        if all_features.ndim == 1 and all_features.shape[0] % 47 == 0:
            num_frames = all_features.shape[0] // 47
            all_features = all_features.reshape((num_frames, 47))
        else:
            return None

    # 위치 변화 크기 계산
    if len(all_features) > 1:
        for i in range(1, len(all_features)):
            diff = all_features[i, :-1] - all_features[i - 1, :-1]
            motion_magnitude = np.linalg.norm(diff)
            all_features[i, -1] = motion_magnitude

    return all_features


def preprocess_features(features, motion_threshold=0.5):
    # 움직임이 있는 프레임만 선택
    motion_values = features[:, -1]
    valid_indices = np.where(motion_values >= motion_threshold)[0]

    if len(valid_indices) == 0:
        return features

    return features[valid_indices]


def resample_to_fixed_frames(features, target_frames=30):
    # 고정된 프레임 수로 리샘플링
    current_frames = len(features)

    if current_frames == target_frames:
        return features

    indices = np.linspace(0, current_frames - 1, target_frames)
    resampled = np.array(
        [
            np.interp(indices, np.arange(current_frames), features[:, i])
            for i in range(features.shape[1])
        ]
    ).T

    return resampled


def process_single_video(args):
    # 단일 비디오 처리 (멀티프로세싱)
    video_path, base_video_dir, base_feature_dir = args

    try:
        # 출력 경로 생성
        rel_path = os.path.relpath(video_path, base_video_dir)
        rel_base, _ = os.path.splitext(rel_path)
        final_save_path = os.path.join(base_feature_dir, rel_base + ".npy")
        save_dir = os.path.dirname(final_save_path)

        # 출력 디렉토리 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 이미 처리된 파일 건너뛰기
        if os.path.exists(final_save_path):
            return f"Skipped: {final_save_path}"

        # Feature 추출
        features = extract_landmarks(video_path)

        if features is None:
            return f"Failed: {video_path}"

        # 전처리
        filtered_features = preprocess_features(features, motion_threshold=0.5)

        # 리샘플링
        final_features = resample_to_fixed_frames(filtered_features, target_frames=30)

        # 최종 저장
        np.save(final_save_path, final_features)
        return f"Success: {final_save_path}"

    except Exception as e:
        return f"Error: {video_path} - {str(e)}"


if __name__ == "__main__":
    base_video_dir = "data/KSL_ACTION_VIDEO"
    base_feature_dir = "features"

    # 비디오 파일 검색
    video_search_path = os.path.join(base_video_dir, "*", "*.MP4")
    video_paths = glob.glob(video_search_path)
    video_paths.extend(glob.glob(os.path.join(base_video_dir, "*", "*.mp4")))

    if not video_paths:
        print(f"Error: No videos found in {base_video_dir}")
    else:
        print(f"Found {len(video_paths)} videos. Starting processing...")

        # 멀티프로세싱 사용
        num_processes = max(1, cpu_count() - 1)  # CPU 코어 수 - 1
        print(f"Using {num_processes} processes")

        # 인자 준비
        args_list = [(vp, base_video_dir, base_feature_dir) for vp in video_paths]

        # 병렬 처리
        with Pool(num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(process_single_video, args_list),
                    total=len(video_paths),
                    desc="Processing videos",
                )
            )

        # 결과 요약
        success_count = sum(1 for r in results if r.startswith("Success"))
        skipped_count = sum(1 for r in results if r.startswith("Skipped"))
        failed_count = sum(
            1 for r in results if r.startswith("Failed") or r.startswith("Error")
        )

        print(f"\n=== Processing Complete ===")
        print(f"Success: {success_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Failed: {failed_count}")
        print(f"Features saved in {base_feature_dir}")
