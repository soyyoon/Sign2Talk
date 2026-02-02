# !pip uninstall -y mediapipe
# !pip install mediapipe==0.10.14

import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter1d
import json
from google.colab import drive
drive.mount('/content/drive')

# --- 설정 ---
VIDEO_DIR = '/content/drive/MyDrive/S2T/KSL_ACTION_VIDEO'
OUTPUT_DIR = '/content/drive/MyDrive/S2T/features_complete_53dim'
TARGET_FRAMES = 30
TARGET_DIM = 53

# 품질 검증 파라미터
MIN_VALID_FRAMES = 20  # 30프레임 중 최소 20프레임은 유효해야 함
MIN_NONZERO_RATIO = 0.3  # 최소 30%는 0이 아닌 값이어야 함

# 동작 구간 감지 파라미터
MOTION_THRESHOLD_PERCENTILE = 30
MIN_MOTION_FRAMES = 10

# 증강 옵션
ENABLE_MIRROR_AUGMENTATION = True  # 좌우 반전 증강 사용 여부

mp_holistic = mp.solutions.holistic

"""### 유틸리티 함수"""

def calculate_distance(p1, p2):
    """유클리드 거리"""
    return float(np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2)))


def calculate_angle(v1, v2):
    """두 벡터 간 각도"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    return float(np.arccos(cos_angle))

"""### Feature 추출"""

def get_53_features(results):
    """
    53차원 피처 추출

    구조:
    - 0-9: 왼손 (10)
    - 10-19: 오른손 (10)
    - 20-34: 양손 상호작용 (15)
    - 35-46: 포즈 (12)
    - 47-52: 얼굴-손 거리 (6)
    """
    feats = []

    # 1. 왼손 (10개)
    if results.left_hand_landmarks:
        lm = results.left_hand_landmarks.landmark
        wrist, tips = 0, [4, 8, 12, 16, 20]

        for tip in tips:
            v1 = [lm[wrist].x - lm[tip].x, lm[wrist].y - lm[tip].y]
            feats.append(calculate_angle(v1, [0, -1]))

        for i in range(len(tips) - 1):
            feats.append(calculate_distance(
                [lm[tips[i]].x, lm[tips[i]].y],
                [lm[tips[i + 1]].x, lm[tips[i + 1]].y]
            ))

        feats.append(0.0)
    else:
        feats.extend([0.0] * 10)

    # 2. 오른손 (10개)
    if results.right_hand_landmarks:
        lm = results.right_hand_landmarks.landmark
        wrist, tips = 0, [4, 8, 12, 16, 20]

        for tip in tips:
            v1 = [lm[wrist].x - lm[tip].x, lm[wrist].y - lm[tip].y]
            feats.append(calculate_angle(v1, [0, -1]))

        for i in range(len(tips) - 1):
            feats.append(calculate_distance(
                [lm[tips[i]].x, lm[tips[i]].y],
                [lm[tips[i + 1]].x, lm[tips[i + 1]].y]
            ))

        feats.append(0.0)
    else:
        feats.extend([0.0] * 10)

    # 3. 양손 상호작용 (15개)
    if results.left_hand_landmarks and results.right_hand_landmarks:
        l_lm = results.left_hand_landmarks.landmark
        r_lm = results.right_hand_landmarks.landmark

        for i in [0, 4, 8, 12, 16, 20]:
            feats.append(calculate_distance(
                [l_lm[i].x, l_lm[i].y],
                [r_lm[i].x, r_lm[i].y]
            ))

        feats.append(calculate_distance(
            [l_lm[0].x, l_lm[0].y],
            [r_lm[0].x, r_lm[0].y]
        ))

        for hand in [l_lm, r_lm]:
            d = [hand[9].x - hand[0].x, hand[9].y - hand[0].y]
            for r in [[0, -1], [1, 0], [0, 1], [-1, 0]]:
                feats.append(calculate_angle(d, r))
    else:
        feats.extend([0.0] * 15)

    # 4. 포즈 (12개)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        for s, e, w in [(11, 13, 15), (12, 14, 16)]:
            v1 = [lm[e].x - lm[s].x, lm[e].y - lm[s].y]
            v2 = [lm[w].x - lm[e].x, lm[w].y - lm[e].y]
            feats.append(calculate_angle(v1, v2))

        sx, sy = (lm[11].x + lm[12].x) / 2, (lm[11].y + lm[12].y) / 2
        hx, hy = (lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2

        for w in [15, 16]:
            feats.extend([
                lm[w].x - sx, lm[w].y - sy,
                lm[w].x - hx, lm[w].y - hy
            ])

        feats.append(calculate_distance([lm[11].x, lm[11].y], [lm[12].x, lm[12].y]))
        feats.append(calculate_distance([sx, sy], [hx, hy]))
    else:
        feats.extend([0.0] * 12)

    # 5. 얼굴-손 거리 (6개)
    if results.face_landmarks:
        f_lm = results.face_landmarks.landmark

        if results.right_hand_landmarks:
            r_idx = results.right_hand_landmarks.landmark[8]
            for t in [1, 13, 263]:
                feats.append(calculate_distance(
                    [r_idx.x, r_idx.y],
                    [f_lm[t].x, f_lm[t].y]
                ))
        else:
            feats.extend([0.0] * 3)

        if results.left_hand_landmarks:
            l_idx = results.left_hand_landmarks.landmark[8]
            for t in [1, 13, 33]:
                feats.append(calculate_distance(
                    [l_idx.x, l_idx.y],
                    [f_lm[t].x, f_lm[t].y]
                ))
        else:
            feats.extend([0.0] * 3)
    else:
        feats.extend([0.0] * 6)

    if len(feats) < TARGET_DIM:
        feats.extend([0.0] * (TARGET_DIM - len(feats)))

    return np.array(feats[:TARGET_DIM], dtype=np.float32)

def get_shoulder_width(results):
    """어깨 너비 추출 (공간 정규화용)"""
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        width = calculate_distance(
            [lm[11].x, lm[11].y],
            [lm[12].x, lm[12].y]
        )
        # 너무 작으면 기본값 사용
        return max(width, 0.05)
    return 0.2  # 기본값

"""### 전처리 함수"""

def spatial_normalization(features_sequence, shoulder_widths):
    """
    공간 정규화: 어깨 너비로 모든 거리 feature 스케일링

    거리 feature 위치:
    - 왼손 거리: 5-9
    - 오른손 거리: 15-19
    - 양손 상호작용 거리: 20-26
    - 포즈 거리: 45-46
    - 얼굴-손 거리: 47-52
    """
    normalized = features_sequence.copy()

    # 거리 feature 인덱스
    distance_indices = (
        list(range(5, 10)) +      # 왼손
        list(range(15, 20)) +     # 오른손
        list(range(20, 27)) +     # 양손 상호작용
        [45, 46] +                # 포즈
        list(range(47, 53))       # 얼굴-손
    )

    for i in range(len(normalized)):
        if shoulder_widths[i] > 0:
            normalized[i, distance_indices] /= shoulder_widths[i]

    return normalized

def interpolate_missing_frames(features_sequence):
    """
    Missing 값 보간 (0인 프레임을 이전/다음 프레임으로 채움)
    """
    interpolated = features_sequence.copy()

    for i in range(len(interpolated)):
        if np.all(interpolated[i] == 0):
            # 이전과 다음 프레임의 평균으로 보간
            if i > 0 and i < len(interpolated) - 1:
                interpolated[i] = (interpolated[i-1] + interpolated[i+1]) / 2
            elif i > 0:
                interpolated[i] = interpolated[i-1]
            elif i < len(interpolated) - 1:
                interpolated[i] = interpolated[i+1]

    return interpolated

def remove_outliers(features_sequence):
    """
    통계적 이상치 제거 (Z-score > 3인 값 클리핑)
    """
    mean = np.mean(features_sequence, axis=0, keepdims=True)
    std = np.std(features_sequence, axis=0, keepdims=True) + 1e-6

    z_scores = np.abs((features_sequence - mean) / std)
    mask = z_scores > 3

    cleaned = features_sequence.copy()
    cleaned[mask] = np.repeat(mean, len(features_sequence), axis=0)[mask]

    return cleaned

def detect_motion_region(features_sequence):
    """동작 구간 감지"""
    if len(features_sequence) < 2:
        return 0, len(features_sequence)

    motion_scores = []
    for i in range(len(features_sequence) - 1):
        diff = np.linalg.norm(features_sequence[i + 1] - features_sequence[i])
        motion_scores.append(diff)

    motion_scores = np.array(motion_scores)
    smoothed = gaussian_filter1d(motion_scores, sigma=1.5)

    threshold = np.percentile(smoothed, MOTION_THRESHOLD_PERCENTILE)
    motion_mask = smoothed > threshold

    if not motion_mask.any():
        return 0, len(features_sequence)

    start_idx = np.argmax(motion_mask)
    end_idx = len(motion_mask) - np.argmax(motion_mask[::-1])

    if end_idx - start_idx < MIN_MOTION_FRAMES:
        center = (start_idx + end_idx) // 2
        start_idx = max(0, center - MIN_MOTION_FRAMES // 2)
        end_idx = min(len(features_sequence), start_idx + MIN_MOTION_FRAMES)

    return start_idx, end_idx

def quality_check(features_sequence):
    """
    품질 검증

    Returns:
        (bool, str): (통과 여부, 실패 이유)
    """
    # 1. 유효 프레임 비율
    valid_frames = np.sum(np.any(features_sequence != 0, axis=1))
    if valid_frames < MIN_VALID_FRAMES:
        return False, f"유효 프레임 부족 ({valid_frames}/{TARGET_FRAMES})"

    # 2. 0이 아닌 값 비율
    nonzero_ratio = np.count_nonzero(features_sequence) / features_sequence.size
    if nonzero_ratio < MIN_NONZERO_RATIO:
        return False, f"0 값 비율 과다 ({nonzero_ratio:.1%})"

    # 3. 극단적 이상치 검사
    if np.max(np.abs(features_sequence)) > 100:
        return False, "극단적 이상치 감지"

    return True, "OK"

def create_mirror_augmentation(features):
    """
    좌우 반전 증강

    Feature 구조:
    - 0-9: 왼손 → 10-19 (오른손)
    - 10-19: 오른손 → 0-9 (왼손)
    - 20-34: 양손 상호작용 (유지)
    - 35-46: 포즈 (좌우 반전)
    - 47-49: 오른손-얼굴 → 50-52 (왼손-얼굴)
    - 50-52: 왼손-얼굴 → 47-49 (오른손-얼굴)
    """
    mirrored = features.copy()

    # 왼손 ↔ 오른손
    left_hand = mirrored[:, 0:10].copy()
    right_hand = mirrored[:, 10:20].copy()
    mirrored[:, 0:10] = right_hand
    mirrored[:, 10:20] = left_hand

    # 포즈 (팔 각도): 왼팔 ↔ 오른팔
    # 35: 왼팔 각도, 36: 오른팔 각도
    left_arm = mirrored[:, 35].copy()
    mirrored[:, 35] = mirrored[:, 36]
    mirrored[:, 36] = left_arm

    # 포즈 (팔 위치): x좌표 반전
    # 37-40: 왼손목, 41-44: 오른손목
    mirrored[:, [37, 39, 41, 43]] *= -1  # x좌표 반전

    # 얼굴-손 거리
    face_right = mirrored[:, 47:50].copy()
    face_left = mirrored[:, 50:53].copy()
    mirrored[:, 47:50] = face_left
    mirrored[:, 50:53] = face_right

    return mirrored

"""### 메인 추출 함수"""

def extract_complete_features(video_path, target_frames=TARGET_FRAMES):
    """
    완전한 전처리 파이프라인

    1. 피처 추출
    2. 품질 검증
    3. 속도 정규화
    4. 공간 정규화
    5. Missing 값 보간
    6. 이상치 제거

    Returns:
        dict: {
            'original': 원본 피처,
            'mirrored': 미러링 피처 (옵션),
            'quality': 품질 정보
        }
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None

    # 1. MediaPipe로 피처 추출
    all_features = []
    shoulder_widths = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for frame in frames:
            results = holistic.process(frame)
            all_features.append(get_53_features(results))
            shoulder_widths.append(get_shoulder_width(results))

    all_features = np.array(all_features)
    shoulder_widths = np.array(shoulder_widths)

    # 2. 동작 구간 감지 (속도 정규화)
    start_idx, end_idx = detect_motion_region(all_features)
    motion_features = all_features[start_idx:end_idx]
    motion_shoulders = shoulder_widths[start_idx:end_idx]

    if len(motion_features) == 0:
        motion_features = all_features
        motion_shoulders = shoulder_widths

    # 3. 리샘플링 (속도 정규화)
    if len(motion_features) >= target_frames:
        indices = np.linspace(0, len(motion_features) - 1, target_frames, dtype=int)
        resampled = motion_features[indices]
        resampled_shoulders = motion_shoulders[indices]
    else:
        indices = np.linspace(0, len(motion_features) - 1, target_frames)
        resampled = np.zeros((target_frames, TARGET_DIM), dtype=np.float32)
        resampled_shoulders = np.zeros(target_frames, dtype=np.float32)

        for dim in range(TARGET_DIM):
            resampled[:, dim] = np.interp(indices, np.arange(len(motion_features)), motion_features[:, dim])

        resampled_shoulders = np.interp(indices, np.arange(len(motion_shoulders)), motion_shoulders)

    # 4. 공간 정규화
    normalized = spatial_normalization(resampled, resampled_shoulders)

    # 5. Missing 값 보간
    interpolated = interpolate_missing_frames(normalized)

    # 6. 이상치 제거
    cleaned = remove_outliers(interpolated)

    # 7. 품질 검증
    is_valid, reason = quality_check(cleaned)

    result = {
        'original': cleaned,
        'quality': {'valid': is_valid, 'reason': reason},
        'motion_region': (start_idx, end_idx, len(all_features))
    }

    # 8. 미러링 증강 (옵션)
    if ENABLE_MIRROR_AUGMENTATION and is_valid:
        result['mirrored'] = create_mirror_augmentation(cleaned)

    return result

"""### 배치 처리"""

def process_all_videos(video_dir, output_dir, overwrite=False):
    """모든 비디오 전처리"""
    os.makedirs(output_dir, exist_ok=True)

    class_folders = sorted([
        d for d in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, d))
        and not d.startswith('.')
    ])

    print("=" * 60)
    print("전처리 파이프라인")
    print("=" * 60)
    print(f"클래스 수: {len(class_folders)}")
    print(f"목표 프레임: {TARGET_FRAMES}")
    print(f"Feature 차원: {TARGET_DIM}")
    print(f"속도 정규화: ON")
    print(f"공간 정규화: ON")
    print(f"품질 검증: ON (최소 {MIN_VALID_FRAMES}프레임)")
    print(f"미러링 증강: {'ON' if ENABLE_MIRROR_AUGMENTATION else 'OFF'}")
    print("=" * 60)

    total_processed = 0
    total_augmented = 0
    total_failed = 0
    failed_reasons = {}

    for class_name in tqdm(class_folders, desc="Processing"):
        class_input_dir = os.path.join(video_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        video_files = (
            glob.glob(os.path.join(class_input_dir, "*.mp4")) +
            glob.glob(os.path.join(class_input_dir, "*.MP4")) +
            glob.glob(os.path.join(class_input_dir, "*.avi")) +
            glob.glob(os.path.join(class_input_dir, "*.AVI"))
        )

        for video_path in video_files:
            video_name = os.path.basename(video_path)
            base_name = os.path.splitext(video_name)[0]

            output_path = os.path.join(class_output_dir, base_name + ".npy")
            mirror_path = os.path.join(class_output_dir, base_name + "_mirror.npy")

            if os.path.exists(output_path) and not overwrite:
                total_processed += 1
                if os.path.exists(mirror_path):
                    total_augmented += 1
                continue

            result = extract_complete_features(video_path, TARGET_FRAMES)

            if result is None or not result['quality']['valid']:
                total_failed += 1
                reason = result['quality']['reason'] if result else "추출 실패"
                failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
                continue

            # 원본 저장
            np.save(output_path, result['original'])
            total_processed += 1

            # 미러링 저장
            if 'mirrored' in result:
                np.save(mirror_path, result['mirrored'])
                total_augmented += 1

    print("\n" + "=" * 60)
    print("전처리 완료!")
    print("=" * 60)
    print(f"   원본: {total_processed}개")
    if ENABLE_MIRROR_AUGMENTATION:
        print(f"   증강: {total_augmented}개 (미러링)")
        print(f"   총합: {total_processed + total_augmented}개")
    print(f"   실패: {total_failed}개")

    if failed_reasons:
        print("\n실패 원인:")
        for reason, count in sorted(failed_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {reason}: {count}개")

    print("=" * 60)

    return {
        'processed': total_processed,
        'augmented': total_augmented,
        'failed': total_failed,
        'failed_reasons': failed_reasons
    }

def analyze_dataset(output_dir):
    """데이터셋 분석"""
    print("\n" + "=" * 60)
    print("데이터셋 분석")
    print("=" * 60)

    class_folders = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]

    class_counts = {}
    for class_name in class_folders:
        class_path = os.path.join(output_dir, class_name)
        count = len(glob.glob(os.path.join(class_path, "*.npy")))
        class_counts[class_name] = count

    if not class_counts:
        print("데이터를 찾을 수 없습니다.")
        return

    counts = list(class_counts.values())
    print(f"\n클래스별 샘플 수:")
    print(f"   평균: {np.mean(counts):.1f}")
    print(f"   중앙값: {np.median(counts):.0f}")
    print(f"   최소: {min(counts)} (클래스 {min(class_counts, key=class_counts.get)})")
    print(f"   최대: {max(counts)} (클래스 {max(class_counts, key=class_counts.get)})")
    print(f"   표준편차: {np.std(counts):.1f}")

    # 불균형 경고
    rare_classes = [k for k, v in class_counts.items() if v < 10]
    if rare_classes:
        print(f"\n샘플 10개 미만 클래스 ({len(rare_classes)}개):")
        for cls in rare_classes[:5]:
            print(f"   - {cls}: {class_counts[cls]}개")
        if len(rare_classes) > 5:
            print(f"   ... 외 {len(rare_classes) - 5}개")

    # 샘플 확인
    sample_class = os.path.join(output_dir, class_folders[0])
    npy_files = glob.glob(os.path.join(sample_class, "*.npy"))

    if npy_files:
        sample = np.load(npy_files[0])
        print(f"\n샘플 검증 (클래스 {class_folders[0]}):")
        print(f"   Shape: {sample.shape}")
        print(f"   평균: {np.mean(sample):.4f}")
        print(f"   표준편차: {np.std(sample):.4f}")
        print(f"   0 비율: {(sample == 0).mean():.1%}")

        if sample.shape == (TARGET_FRAMES, TARGET_DIM):
            print("\n완벽합니다! 학습 가능!")
        else:
            print(f"\nShape 오류: 예상 ({TARGET_FRAMES}, {TARGET_DIM})")

    print("=" * 60)

    # 통계 저장
    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({
            'class_counts': class_counts,
            'total_samples': sum(counts),
            'num_classes': len(class_counts),
            'mean_samples_per_class': float(np.mean(counts)),
            'std_samples_per_class': float(np.std(counts))
        }, f, indent=2)
    print(f"\n통계 저장: {stats_path}")

"""### 실행"""

if __name__ == "__main__":
    # 전처리 실행
    results = process_all_videos(
        video_dir=VIDEO_DIR,
        output_dir=OUTPUT_DIR,
        overwrite=True
    )

    # 데이터셋 분석
    analyze_dataset(OUTPUT_DIR)

    print(f"\n완료! 이제 학습하세요!")
    print(f"저장 경로: {OUTPUT_DIR}")

