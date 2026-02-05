import os
import json
import glob
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from google.colab import drive

drive.mount('/content/drive')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 경로
DATA_DIR = '/content/drive/MyDrive/features_complete_53dim'
SAVE_DIR = '/content/drive/MyDrive/SavedModels_53_normalized'

# 데이터 설정
MAX_FRAMES = 30
FEATURE_DIM = 53
BATCH_SIZE = 32
EPOCHS = 100

# 미러링 증강 데이터 사용 여부
USE_MIRRORED_DATA = True

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("=" * 60)
print("학습 설정")
print("=" * 60)
print(f"데이터: {DATA_DIR}")
print(f"저장: {SAVE_DIR}")
print(f"Seed: {SEED}")
print(f"Feature: {FEATURE_DIM}차원 (속도 정규화 완료)")
print(f"미러링 데이터: {'사용' if USE_MIRRORED_DATA else '미사용'}")
print("=" * 60)

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# 라벨 정의
KSL_LABELS = {
    "01": "Hi", "02": "What", "03": "Meet", "04": "Bibimbap", "05": "Glad",
    "06": "Hobby", "07": "Me", "08": "Movie", "09": "Face", "10": "See",
    "11": "Name", "13": "Thank", "14": "Equal", "15": "Sorry",
    "16": "Eat", "17": "Fine", "18": "Do_Effort", "20": "Age",
    "21": "Again", "22": "How_many", "23": "Day", "24": "Good,Nice", "25": "When",
    "26": "We", "27": "Subway", "29": "Bus", "30": "Ride",
    "31": "Cellphone", "32": "Where", "34": "Location",
    "36": "Responsibility", "37": "Who", "38": "Arrive", "39": "Family", "40": "Time",
    "41": "Introduction", "42": "Receive", "43": "Please", "44": "Walk",
    "47": "Sister", "48": "Study", "49": "Human", "50": "Now",
    "51": "Special", "52": "Yesterday", "54": "Test", "55": "End",
    "56": "You", "57": "Worried", "58": "Marry", "59": "Effort", "60": "No",
    "61": "Sweat", "62": "Yet", "63": "Finally", "64": "Born", "65": "Success",
    "66": "Favor", "67": "Seoul", "68": "Evening", "69": "Experience", "70": "Invite",
    "71": "Food", "72": "Want", "74": "One_Hour", "76": "Good", "77": "Care"
}

"""### 데이터 로드"""

def load_all_data(data_dir, max_frames=30, feature_dim=53, use_mirrored=True):
    """
    전처리된 데이터 로드
    """
    print("\n" + "=" * 60)
    print("데이터 로딩 중...")
    print("=" * 60)

    X_data = []
    y_data = []

    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and not d.startswith('.')
    ])

    class_samples = {}
    original_count = 0
    mirrored_count = 0

    for class_idx, class_name in enumerate(tqdm(class_names, desc="Loading")):
        class_path = os.path.join(data_dir, class_name)
        npy_files = glob.glob(os.path.join(class_path, "*.npy"))

        count = 0
        for fpath in npy_files:
            # 미러링 파일 처리
            is_mirrored = "_mirror.npy" in fpath
            if is_mirrored and not use_mirrored:
                continue

            try:
                arr = np.load(fpath)
                if arr.shape == (max_frames, feature_dim):
                    X_data.append(arr)
                    y_data.append(class_idx)
                    count += 1

                    if is_mirrored:
                        mirrored_count += 1
                    else:
                        original_count += 1
                else:
                    print(f"Shape 오류: {fpath} - {arr.shape}")
            except Exception as e:
                print(f"로드 실패: {fpath} - {e}")
                continue

        class_samples[class_name] = count

    # 통계 출력
    print(f"\n로드 완료!")
    print(f"   총 샘플: {len(X_data)}개")
    print(f"   - 원본: {original_count}개")
    if use_mirrored:
        print(f"   - 미러링: {mirrored_count}개")
    print(f"   클래스 수: {len(class_names)}개")

    # 클래스별 샘플 수
    counts = list(class_samples.values())
    print(f"\n클래스별 샘플 수:")
    print(f"   평균: {np.mean(counts):.1f}")
    print(f"   최소: {min(counts)} (클래스 {min(class_samples, key=class_samples.get)})")
    print(f"   최대: {max(counts)} (클래스 {max(class_samples, key=class_samples.get)})")
    print(f"   표준편차: {np.std(counts):.1f}")

    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.int32), class_names

"""### 손실 함수"""

def focal_loss_fn(num_classes, alpha=0.25, gamma=2.0, label_smoothing=0.1):
    """
    Focal Loss: 어려운 샘플에 더 집중
    Label Smoothing: 과적합 방지
    """
    def loss(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p = tf.reduce_sum(y_true * y_pred, axis=-1)
        return tf.reduce_mean(alpha * tf.pow(1 - p, gamma) * ce)
    return loss

"""### 데이터 증강"""

def augment_sequence(sequence):
    """
    학습 시 추가 증강 (경미한 노이즈만)

    주의: 속도 왜곡은 하지 않음! (이미 전처리에서 속도 정규화함)
    """
    aug = sequence.copy()

    # 1. 미세 노이즈 (5% 확률)
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.01, size=aug.shape)
        aug += noise

    # 2. 스케일링 (5% 확률)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.98, 1.02)
        aug *= scale

    return aug.astype(np.float32)


def create_dataset(X, y, batch_size, augment=True):
    """TensorFlow Dataset 생성"""
    def generator():
        indices = np.arange(len(X))
        while True:
            np.random.shuffle(indices)
            for idx in indices:
                sample = augment_sequence(X[idx]) if augment else X[idx]
                yield sample, y[idx]

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(MAX_FRAMES, FEATURE_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

"""### 모델 아키텍쳐"""

def build_model_v1_speed_invariant(input_shape, num_classes):
    """
    V1: Multi-scale Temporal CNN (속도 불변)

    다양한 dilation rate로 여러 시간 스케일을 동시에 학습
    → 속도 변화에 강건함
    """
    inp = layers.Input(shape=input_shape)

    # Multi-scale convolution
    conv1 = layers.Conv1D(32, 3, dilation_rate=1, padding="same")(inp)
    conv2 = layers.Conv1D(32, 3, dilation_rate=2, padding="same")(inp)
    conv3 = layers.Conv1D(32, 3, dilation_rate=4, padding="same")(inp)

    # 병합
    x = layers.Concatenate()([conv1, conv2, conv3])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Residual block
    res = x
    x = layers.Conv1D(96, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(96, 3, padding="same")(x)
    x = layers.Add()([res, x])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inp, out, name="V1_MultiScale_CNN")


def build_model_v2_cnn_attention(input_shape, num_classes):
    """
    V2: CNN + Attention

    CNN으로 지역 패턴 추출 → Attention으로 중요 부분 강조
    """
    inp = layers.Input(shape=input_shape)

    # CNN layers
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Global Average/Max Pooling 병렬
    gap = layers.GlobalAveragePooling1D()(x)
    gmp = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([gap, gmp])

    # Dense layers
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inp, out, name="V2_CNN_Attention")


def build_model_v3_transformer(input_shape, num_classes):
    """
    V3: Transformer

    Self-attention으로 시퀀스 전체의 관계 학습
    """
    inp = layers.Input(shape=input_shape)

    # Projection
    x = layers.Dense(64)(inp)

    # Transformer block
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward network
    ffn = models.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64),
    ])
    ffn_output = ffn(x)
    x = layers.Add()([x, ffn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inp, out, name="V3_Transformer")


def build_model_v4_hybrid(input_shape, num_classes):
    """
    V4: Hybrid (CNN + Transformer)

    CNN으로 지역 패턴 추출 → Transformer로 시퀀스 관계 학습
    """
    inp = layers.Input(shape=input_shape)

    # CNN feature extraction
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)  # 30 → 15

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Transformer
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inp, out, name="V4_Hybrid_CNN_Transformer")

"""### 앙상블"""

def weighted_ensemble_tta(models_list, X_norm, weights, n_tta=20):
    """
    TTA (Test Time Augmentation) + 가중 앙상블
    """
    num_classes = models_list[0].output_shape[-1]
    final_probs = np.zeros((len(X_norm), num_classes))

    for i, model in enumerate(models_list):
        print(f"  📦 {model.name} (가중치 {weights[i]:.2f}) 투표 중...")
        m_probs = []

        for t in range(n_tta):
            if t == 0:
                X_in = X_norm
            else:
                X_in = np.array([augment_sequence(s) for s in X_norm])
            m_probs.append(model.predict(X_in, verbose=0))

        final_probs += np.mean(m_probs, axis=0) * weights[i]

    return final_probs


def find_best_ensemble_weights(models_list, X_val, y_val, n_tta=15):
    """
    최적의 앙상블 가중치 찾기 (Grid Search)
    """
    print("\n🔍 최적 앙상블 가중치 탐색 중...")

    # 각 모델의 TTA 확률 미리 계산
    all_model_probs = []
    for model in models_list:
        print(f"  > {model.name} TTA 확률 계산 중...")
        m_probs = []
        for t in range(n_tta):
            X_in = X_val if t == 0 else np.array([augment_sequence(s) for s in X_val])
            m_probs.append(model.predict(X_in, verbose=0))
        all_model_probs.append(np.mean(m_probs, axis=0))

    best_acc = 0
    best_weights = None

    # 가중치 탐색 (0.1 단위)
    print("  > Grid search 중...")
    weights_candidates = []

    if len(models_list) == 3:
        for w1 in np.linspace(0, 0.7, 8):
            for w2 in np.linspace(0, 0.7, 8):
                for w3 in np.linspace(0, 0.7, 8):
                    if abs(w1 + w2 + w3 - 1.0) < 0.01:
                        weights_candidates.append((w1, w2, w3))
    elif len(models_list) == 4:
        for w1 in np.linspace(0, 0.6, 7):
            for w2 in np.linspace(0, 0.6, 7):
                for w3 in np.linspace(0, 0.6, 7):
                    for w4 in np.linspace(0, 0.6, 7):
                        if abs(w1 + w2 + w3 + w4 - 1.0) < 0.01:
                            weights_candidates.append((w1, w2, w3, w4))

    for w in tqdm(weights_candidates, desc="  Testing weights"):
        final_probs = sum(all_model_probs[i] * w[i] for i in range(len(models_list)))
        preds = np.argmax(final_probs, axis=1)
        acc = accuracy_score(y_val, preds)

        if acc > best_acc:
            best_acc = acc
            best_weights = w

    print(f"\n최적 가중치: {[f'{w:.2f}' for w in best_weights]}")
    print(f"   검증 정확도: {best_acc*100:.2f}%")

    return best_weights, all_model_probs

"""### 메인 실행"""

if __name__ == "__main__":
    # 1. 데이터 로드
    X_raw, y_raw, class_names = load_all_data(
        DATA_DIR,
        max_frames=MAX_FRAMES,
        feature_dim=FEATURE_DIM,
        use_mirrored=USE_MIRRORED_DATA
    )

    NUM_CLASSES = len(class_names)

    # 2. Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw,
        test_size=0.15,
        stratify=y_raw,
        random_state=SEED
    )

    print(f"\n데이터 분할:")
    print(f"   Train: {len(X_train)}개")
    print(f"   Test: {len(X_test)}개")

    # 3. 정규화 (Z-score)
    print(f"\n정규화 중...")
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    print(f"   Train 평균: {X_train_norm.mean():.4f}, 표준편차: {X_train_norm.std():.4f}")
    print(f"   Test 평균: {X_test_norm.mean():.4f}, 표준편차: {X_test_norm.std():.4f}")

    # 4. 모델 학습
    builders = [
        build_model_v1_speed_invariant,
        build_model_v2_cnn_attention,
        build_model_v3_transformer,
        build_model_v4_hybrid
    ]

    trained_models = []

    print("\n" + "=" * 60)
    print("모델 학습 시작")
    print("=" * 60)

    for i, builder in enumerate(builders):
        print(f"\n[{i+1}/{len(builders)}] {builder.__name__} 학습 중...")

        model = builder((MAX_FRAMES, FEATURE_DIM), NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
            loss=focal_loss_fn(NUM_CLASSES),
            metrics=['accuracy']
        )

        # Dataset 생성
        train_ds = create_dataset(X_train_norm, y_train, BATCH_SIZE, augment=True)
        val_ds = tf.data.Dataset.from_tensor_slices((X_test_norm, y_test)).batch(BATCH_SIZE)

        # 학습
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    mode='max'
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ],
            verbose=1
        )

        # 저장
        model_path = os.path.join(SAVE_DIR, f"model_v{i+1}.keras")
        model.save(model_path)
        print(f"저장: {model_path}")

        # 개별 성능
        test_loss, test_acc = model.evaluate(val_ds, verbose=0)
        print(f"   Test Accuracy: {test_acc*100:.2f}%")

        trained_models.append(model)

    # 5. 앙상블 최적화
    print("\n" + "=" * 60)
    print("앙상블 최적화")
    print("=" * 60)

    best_weights, all_probs = find_best_ensemble_weights(
        trained_models,
        X_test_norm,
        y_test,
        n_tta=20
    )

    # 6. 최종 예측
    final_probs = sum(all_probs[i] * best_weights[i] for i in range(len(trained_models)))
    final_preds = np.argmax(final_probs, axis=1)
    final_acc = accuracy_score(y_test, final_preds)

    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"앙상블 정확도: {final_acc*100:.2f}%")

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
    plt.title(f'Confusion Matrix (Acc: {final_acc*100:.2f}%)', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.tight_layout()

    cm_path = os.path.join(SAVE_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion Matrix 저장: {cm_path}")
    plt.show()

    # 8. Classification Report (상위/하위 10개 클래스)
    report = classification_report(
        y_test,
        final_preds,
        target_names=[class_names[i] for i in range(NUM_CLASSES)],
        output_dict=True
    )

    # F1-score 기준 정렬
    class_f1 = [(class_names[i], report[class_names[i]]['f1-score'])
                for i in range(NUM_CLASSES)]
    class_f1.sort(key=lambda x: x[1], reverse=True)

    print("\n성능 우수 클래스 (Top 10):")
    for name, f1 in class_f1[:10]:
        print(f"   {name}: F1={f1:.3f}")

    print("\n성능 부족 클래스 (Bottom 10):")
    for name, f1 in class_f1[-10:]:
        print(f"   {name}: F1={f1:.3f}")

    # 9. 저장
    np.save(os.path.join(SAVE_DIR, 'mean.npy'), mean)
    np.save(os.path.join(SAVE_DIR, 'std.npy'), std)
    np.save(os.path.join(SAVE_DIR, 'ensemble_weights.npy'), np.array(best_weights))

    with open(os.path.join(SAVE_DIR, 'class_names.json'), 'w') as f:
        json.dump(class_names, f, indent=2)

    with open(os.path.join(SAVE_DIR, 'training_results.json'), 'w') as f:
        json.dump({
            'final_accuracy': float(final_acc),
            'ensemble_weights': [float(w) for w in best_weights],
            'num_classes': NUM_CLASSES,
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'feature_dim': FEATURE_DIM,
            'max_frames': MAX_FRAMES,
            'classification_report': report
        }, f, indent=2)

    print(f"\n모든 파일 저장 완료: {SAVE_DIR}")
    print("=" * 60)

