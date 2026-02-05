# 한국 수어 인식 시스템 (Korean Sign Language Recognition)

실시간 웹캠 기반 한국 수어 단어 인식 및 GPT 문장 생성 시스템

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [데이터 전처리](#데이터-전처리)
3. [모델 학습](#모델-학습)
4. [실시간 인식](#실시간-인식)
5. [설치 및 실행](#설치-및-실행)
6. [문제 해결](#문제-해결)

---

## 프로젝트 개요

### 시스템 구성
```
데이터 수집 (1,210개 영상)
    ↓
전처리 파이프라인 (속도/공간 정규화)
    ↓
모델 학습 (4개 앙상블, 96.69% 정확도)
    ↓
실시간 웹캠 인식 + GPT 문장 생성
```

### 주요 특징
- ✅ **속도 불변 인식**: 사람마다 다른 수어 속도에 강건
- ✅ **공간 정규화**: 체격 차이 자동 보정
- ✅ **4개 모델 앙상블**: 96.69% 테스트 정확도
- ✅ **실시간 연속 인식**: 5프레임마다 자동 예측
- ✅ **GPT 문장 생성**: 단어 리스트 → 자연스러운 문장

### 성능
| 지표 | 값 |
|------|-----|
| 테스트 정확도 | 96.69% |
| 클래스 수 | 67개 |
| 학습 데이터 | 2,420개 (원본 1,210 + 미러링 1,210) |
| 실시간 FPS | 25-30 |
| 평균 응답 시간 | 0.2초 (단어), 2초 (문장) |

---

## 데이터 전처리

### 파일: `data_preprocessing.py`

### 6가지 핵심 전처리

#### 1. 속도 정규화 
```python
문제: 사람마다 수어 속도가 다름
해결: 실제 동작 구간만 추출 → 30프레임으로 리샘플링
효과: 속도가 다른 같은 수어를 동일하게 인식
```

**동작 원리:**
- 프레임 간 L2 norm 계산
- Gaussian smoothing (σ=1.5)
- Dynamic threshold (30th percentile)
- Motion region만 추출 후 리샘플링

#### 2. 공간 정규화 
```python
문제: 사람마다 체격이 다름
해결: 어깨 너비로 모든 거리 feature 정규화
효과: 체격 차이 보정, 동작 자체만 학습
```

**정규화 대상:**
- 손 내 거리 (finger tips)
- 양손 간 거리
- 손-얼굴 거리
- 포즈 거리

#### 3. 품질 검증
```python
문제: MediaPipe가 랜드마크를 제대로 감지하지 못하는 경우
해결: 3가지 검증 통과해야 저장
```

**검증 기준:**
- ✅ 유효 프레임 ≥ 20/30 (66%)
- ✅ 0이 아닌 값 ≥ 30%
- ✅ 극단적 이상치 없음 (|값| < 100)

**결과:** 불량 데이터 자동 필터링 (19개 제거, 1.5%)

#### 4. Missing 값 보간
```python
문제: 일부 프레임만 감지 실패
해결: 이전/다음 프레임 평균으로 채움
효과: 일시적 감지 실패를 자연스럽게 복구
```

#### 5. 이상치 제거
```python
문제: MediaPipe 오감지
해결: Z-score 기반 클리핑 (threshold=3σ)
효과: 극단적 오류값 제거
```

#### 6. 미러링 증강
```python
문제: 데이터 부족, 좌우 편향
해결: 좌우 반전 (left hand ↔ right hand)
효과: 데이터 2배 + 좌우 방향 불변성
```

### 전처리 결과
```
입력: 1,210개 영상 (67개 클래스)
출력: 2,420개 샘플 (원본 1,210 + 미러 1,210)
실패: 19개 (품질 검증 실패)
Shape: (30, 53)
```

### Feature 구성 (53차원)
```
왼손:     10차원 (각도 5 + 거리 5)
오른손:   10차원 (각도 5 + 거리 5)
양손:     15차원 (거리 7 + 방향 8)
포즈:     12차원 (각도 2 + 위치 8 + 거리 2)
얼굴-손:   6차원 (거리 6)
───────────────────────────────
합계:     53차원
```

### 실행
```bash
cd /Users/soyun/Desktop/s2t/26.02.02
python data_preprocessing.py
```

**출력:** `/content/drive/MyDrive/S2T/features_complete_53dim/*.npy`

---

## 모델 학습

### 파일: `training_model.py`

### 데이터 분할
```
총 데이터: 2,420개
Train: 2,057개 (85%)
Test:    363개 (15%)
분할 방식: Stratified (클래스 비율 유지)
```

### 4개 모델 아키텍처

#### V1: Multi-scale Temporal CNN (속도 불변)
```
입력 (30, 53)
  ↓
Parallel Conv1D (dilation=1,2,4)  ← 다양한 시간 스케일
  ↓
Concat (96)
  ↓
Residual Block (96)
  ↓
GlobalAveragePooling
  ↓
Dense(128) + Dropout(0.5)
  ↓
Softmax(67)

파라미터: ~120K
특징: 속도 변화에 가장 강건
```

#### V2: CNN + Attention
```
입력 (30, 53)
  ↓
Conv1D(64) + BatchNorm
  ↓
Conv1D(128) + BatchNorm
  ↓
GAP + GMP (병렬)  ← 중요 feature 강조
  ↓
Concat + Dense(128) + Dropout(0.5)
  ↓
Softmax(67)

파라미터: ~100K
특징: Global Average/Max Pooling 병렬
```

#### V3: Transformer
```
입력 (30, 53)
  ↓
Dense(64) - Projection
  ↓
MultiHeadAttention (heads=4)  ← 시퀀스 관계 학습
  ↓
LayerNorm + Residual
  ↓
FFN (64→128→64)
  ↓
GlobalAveragePooling
  ↓
Softmax(67)

파라미터: ~150K
특징: Self-attention으로 장거리 의존성 포착
```

#### V4: Hybrid (CNN + Transformer)
```
입력 (30, 53)
  ↓
Conv1D(64) + MaxPooling → (15, 64)  ← 지역 패턴
  ↓
Conv1D(128) → (15, 128)
  ↓
MultiHeadAttention (heads=4)  ← 시퀀스 관계
  ↓
GlobalAveragePooling
  ↓
Softmax(67)

파라미터: ~180K
특징: CNN + Transformer 장점 결합
```

### 학습 전략

#### 1. Loss Function
```python
Focal Loss (alpha=0.25, gamma=2.0)
  → 어려운 샘플에 집중
  → 클래스 불균형 완화

Label Smoothing (0.1)
  → 과적합 방지
  → 모델의 과한 확신 억제
```

#### 2. Optimizer
```python
AdamW (weight_decay=1e-4)
초기 LR: 0.001
ReduceLROnPlateau: 5 epoch patience
EarlyStopping: 15 epoch patience (val_accuracy)
```

#### 3. 데이터 증강 (학습 시)
```python
Gaussian noise: σ=0.01 (50% 확률)
Scaling: 0.98~1.02배 (50% 확률)
```

#### 4. 정규화
```python
mean = X_train.mean(axis=(0,1))
std = X_train.std(axis=(0,1))
X_normalized = (X - mean) / std
```

#### 5. TTA (Test Time Augmentation)
```python
각 샘플을 20번 증강해서 예측
최종 예측 = mean(predictions)
```

#### 6. 앙상블
```python
Grid Search로 최적 가중치 탐색
V1: 0.10
V2: 0.10
V3: 0.50  ← Transformer가 가장 중요!
V4: 0.30

최종 예측 = Σ(model_i × weight_i)
```

### 학습 결과
```
개별 모델:
  V1 (Multi-scale): ~76%
  V2 (Attention):   ~74%
  V3 (Transformer): ~76%
  V4 (Hybrid):      ~77%

앙상블 (TTA 20x): 96.69% ⭐
```

### 출력 파일
```
models
├── model_v1.keras              # V1 모델
├── model_v2.keras              # V2 모델
├── model_v3.keras              # V3 모델
├── model_v4.keras              # V4 모델
├── mean.npy                    # 정규화 평균
├── std.npy                     # 정규화 표준편차
├── ensemble_weights.npy        # 앙상블 가중치
├── class_names.json            # 클래스 이름
├── training_results.json       # 학습 결과
└── confusion_matrix.png        # Confusion Matrix
```

### 실행
```bash
cd /Users/soyun/Desktop/s2t/26.02.02/models
python training_model.py
```

---

## 실시간 인식

### 버전 1: 단순 단어 인식 (`webcam.py`)

#### 기능
- 실시간 웹캠 수어 인식
- 단어 예측 + 신뢰도 표시
- 최근 예측 히스토리

#### 화면 구성
```
┌──────────────────────────────────────┐
│ Sign Language Recognition            │
│ Buffer: 30/30  |  FPS: 28.5          │
│                                      │
│ 01: 안녕       92.3%                 │
│                                      │
│     [웹캠 영상 + 손 랜드마크]        │
│                                      │
│ Recent:                              │
│ 01:안녕(92%) → 11:이름(88%)          │
│                                      │
│ Press 'Q' to quit | 'R' to reset     │
└──────────────────────────────────────┘
```

#### 키보드
- **Q**: 종료
- **R**: 리셋

#### 설정
```python
CONFIDENCE_THRESHOLD = 0.7   # 신뢰도 임계값
FRAME_STRIDE = 10            # 예측 빈도
DUPLICATE_FILTER_SEC = 2.0   # 중복 방지 시간
```

---

### 버전 2: 연속 인식 + 문장 생성 (`webcam_space.py`) 

#### 기능
- ✅ **실시간 연속 인식**: 5프레임마다 자동 예측
- ✅ **자동 단어 추가**: 신뢰도 70% 이상 자동 추가
- ✅ **중복 방지**: 1.5초 간격
- ✅ **GPT 문장 생성**: Space 2번으로 즉시 생성

#### 사용 흐름
```
Space 1번
  ↓
실시간 연속 인식 (자동)
"01:안녕" 추가
"11:이름" 추가
"02:뭐" 추가
  ↓
Space 2번
  ↓
GPT 문장 생성 (자동)
"안녕하세요! 이름이 뭐예요?"
```

#### 화면 구성
```
┌────────────────────────────────────────┐
│ Sign Language Recognition              │
│ RECORDING          01:안녕 92%  ← 실시간│
│ Buffer: 30/30  |  FPS: 28.5            │
│                                        │
│      [웹캠 + 랜드마크]                 │
│                                        │
│ Words:                                 │
│ 01:안녕 → 11:이름 → 02:뭐  ← 단어 나열 │
│                                        │
│ 문장: 안녕하세요! 이름이 뭐예요?       │
│ Space: Start/Stop | R: Reset           │
└────────────────────────────────────────┘
```

#### 키보드
- **Space**: 녹화 시작/종료 + 문장 생성
- **R**: 리셋
- **Q**: 종료

#### 설정
```python
CONFIDENCE_THRESHOLD = 0.70  # 신뢰도 임계값
FRAME_STRIDE = 5             # 예측 빈도 (더 자주)
중복 방지: 1.5초             # last_prediction_time
```

#### OpenAI API 설정
```bash
# 환경 변수 (권장)
export OPENAI_API_KEY='sk-proj-xxxxx...'

# 또는 코드에 직접 입력
OPENAI_API_KEY = "sk-proj-xxxxx..."
```

---

## 설치 및 실행

### 1. 패키지 설치
```bash
# 기본 패키지
pip install opencv-python mediapipe tensorflow numpy pillow

# GPT 문장 생성 (webcam_realtime.py)
pip install openai
```

### 2. 모델 파일 확인
```bash
cd /Users/soyun/Desktop/s2t/26.02.02/models
ls *.keras *.npy *.json

# 필수 파일 (8개)
# model_v1.keras
# model_v2.keras
# model_v3.keras
# model_v4.keras
# mean.npy
# std.npy
# ensemble_weights.npy
# class_names.json
```

### 3. 실행

#### 단순 인식
```bash
python webcam.py
```

#### 연속 인식 + 문장 생성
```bash
export OPENAI_API_KEY='sk-proj-xxxxx...'
python webcam_space.py
```

---

## 문제 해결

### 설치 관련

#### Q1: MediaPipe 에러
```bash
pip uninstall mediapipe -y
pip install mediapipe==0.10.9
```

#### Q2: Protobuf 에러
```bash
pip install protobuf==3.20.3
```

#### Q3: NumPy 버전 충돌
```bash
pip install "numpy<2"
```

#### Q4: TensorFlow 버전
```bash
pip install tensorflow==2.15.0
```

### 인식 관련

#### Q1: 인식이 안 됨
```python
# 신뢰도 낮추기
CONFIDENCE_THRESHOLD = 0.5

# 예측 빈도 높이기
FRAME_STRIDE = 5

# MediaPipe 감도 낮추기
min_detection_confidence=0.3
```

#### Q2: 같은 단어만 반복
```python
# 중복 방지 시간 늘리기
DUPLICATE_FILTER_SEC = 3.0

# 또는 (webcam_realtime.py)
current_time - self.last_prediction_time > 2.5
```

#### Q3: FPS가 너무 낮음
```python
# 해상도 낮추기
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 예측 빈도 낮추기
FRAME_STRIDE = 15
```

#### Q4: 웹캠이 안 잡힘
```python
# 카메라 번호 변경
cap = cv2.VideoCapture(1)  # 0 → 1 또는 2
```

### GPT 관련

#### Q1: API 키 오류
```bash
# 환경 변수 확인
echo $OPENAI_API_KEY

# 설정
export OPENAI_API_KEY='sk-proj-xxxxx...'

# 영구 설정 (Mac)
echo "export OPENAI_API_KEY='sk-proj-xxxxx...'" >> ~/.zshrc
source ~/.zshrc
```

#### Q2: 문장 생성 안 됨
```python
# 터미널 오류 메시지 확인
# API 키 확인
# 인터넷 연결 확인
```

---

## 성능 최적화

### 빠른 실행 (CPU)
```python
# 해상도 낮추기
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MediaPipe 간소화
model_complexity=0
min_detection_confidence=0.3

# 예측 빈도 낮추기
FRAME_STRIDE = 15
```

### 정확도 우선 (GPU)
```python
# 해상도 높이기
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# MediaPipe 고품질
model_complexity=2
min_detection_confidence=0.7

# 신뢰도 높이기
CONFIDENCE_THRESHOLD = 0.85
```

---

## 프로젝트 구조

```
26.02.02/
├── data_preprocessing.py        # 전처리
├── training_model.py            # 모델 학습
├── webcam_local.py              # 단순 인식
├── webcam_realtime.py           # 연속 인식 + GPT
├── model_v1.keras               # 학습된 모델 1
├── model_v2.keras               # 학습된 모델 2
├── model_v3.keras               # 학습된 모델 3
├── model_v4.keras               # 학습된 모델 4
├── mean.npy                     # 정규화 평균
├── std.npy                      # 정규화 표준편차
├── ensemble_weights.npy         # 앙상블 가중치
├── class_names.json             # 클래스 이름
└── README.md                    # 이 파일
```

---

## 성능 지표

### 모델 성능
```
테스트 정확도: 96.69%
클래스 수: 67개
샘플 수: 2,420개
Confusion Matrix: 거의 완벽한 대각선
```

### 실시간 성능
```
FPS: 25-30
응답 시간: 0.2초 (단어)
문장 생성: 1-3초 (GPT)
신뢰도: 70-85% 이상만 인식
```

### 개선 효과
```
속도 정규화: +10%p
공간 정규화: +5%p
미러링 증강: +8%p
4개 앙상블: +5%p
TTA: +3%p
──────────────────
최종: 96.69%
```

---
