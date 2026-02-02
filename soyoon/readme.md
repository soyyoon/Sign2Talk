## 전처리
### data_preprocessing.py

전처리 내용
1. 속도 정규화
- 사람마다 수어 구현 속도가 다름
- 실제 동작 구간만 추출 -> 30 프레임으로 리샘플링
- 속도가 다른 사람의 같은 수어를 동일하게 인식

2. 공간 정규화
- 사람마다 체격이 다름
- 어깨 너비로 모든 거리 feature 나눔
- 체격 차이를 보정하여 동적 자체만 학습

3. 품질 검증
- MediaPipe가 랜드마크를 제대로 감지하지 못하는 경우
- 3가지 검증 통과해야 저장
    - 유효 프레임 >= 20/30 (66% 이상)
    - 0이 아닌 값 >= 30%
    - 극단적 이상치 없음 (|값| < 100)
- 불량 데이터 자동 필터링 -> 학습 품질 향상

4. Missing 값 보간
- 일부 프레임만 감지 실패
- 이전/다음 프레임 평균으로 채움
- 일시적 감지 실패를 자연스럽게 복구

5. 이상치 제거
- MediaPipe 오감지
- Z-score 기반 클리핑
- 극단적 오류값 제거

6. 미러링 증강
- 데이터 2배 증가 + 좌우 방향 불변성

### feature_complete_53dim


## 모델 학습
### training_model.py

데이터 정보
1. 입력 데이터
- 경로: /content/drive/MyDrive/S2T/features_complete_53dim
- 총 샘플: 2420개
  - 원본: 1210개
  - 미러링: 1210개
- 클래스: 67개
- Shape: (30, 53)

2. 학습/테스트 분할
- Train: 2057개 (85%)
- Test: 363개 (15%)
- Stratified sampling (클래스 비율 유지)

모델 아키텍처 (4개)
1. V1: Multi-scale Temporal CNN
- 입력(30, 53) -> Multi-scale Conv (dilation 1, 2, 4) -> Concat(96) -> Residual Block (96) -> GlobalAveragePooling -> Dense(128) + Dropout(0.5) -> Softmax(67)
- 다양한 속도 패턴 학습
- 속도 변화에 가장 강건

2. V2: CNN + Attention
- 입력(30, 53) -> Conv1D(64) + BatchNorm -> Conv1D(128) + BatchNorm -> Gap + GMP -> Concat -> Dense(128) + Dropout(0.5) -> Softmax(67)
- Global Average / Max Pooling 병렬 사용
- 중요한 feature 강조

3. V3: Transformer
- 입력(30, 53) -> Dense(64)-Projection -> multiHeadAttention(heads=4) -> LayerNorm + Residual -> FFN (64->128->64) -> GlobalAveragePooling -> Softmax(67)
- Self-attention으로 시퀸스 관계 학습
- 장거리 의존성 포착

4. V4: Hybrid (CNN+Transformer)
- 입력(30, 53) -> Conv1D(64) + MaxPooling -> (15, 64) -> Conv1D(128) -> (15, 128) -> MultiHeadAttention(heads = 4) -> GlobalAveragePooling -> Softmax(67)
- CNN으로 지역 패턴 추출 + Transformer로 시퀸스 관계
- 두 방식의 장점 결합

학습 전략
1. Focal Loss +  Label Smoothing
- Focal Loss (alpha=0.25, gamma=2.0)
  - 어려운 샘플에 집중
  - 클래스 불균형 완화

- Label Smoothing (0.1)
  - 과적합 방지
  - 모델의 과한 확신 억제

2. 데이터 증강
- 미세 노이즈: ±1% (50% 확률)
- 스케일링: 0.98~1.02배 (50% 확률)

3. Optimizer & Learning Rate
- AdamW (weight_decay=1e-4)
  - 초기 LR: 0.001
  - ReduceLROnPlateau: 5 epoch patience

- EarlyStopping
  - Monitor: val_accuracy
  - Patience: 15 epochs

4. 정규화
- mean = X_train.mean(axis=(0,1))
- std = X_train.std(axis=(0,1))
- X_normalized = (X - mean) / std


5. TTA (Test Time Augmentation)
- 각 샘플을 20번 증강해서 예측

6. 가중 앙상블
- Grid Search로 최적 가중치 찾기


## 실시간 웹캠

### webcam.py
- only 단어 인식

1. 파일 구조 확인
/Users/soyun/Desktop/s2t/26.02.02/
├── model_v1.keras              # 필수
├── model_v2.keras              # 필수
├── model_v3.keras              # 필수
├── model_v4.keras              # 필수
├── mean.npy                    # 필수
├── std.npy                     # 필수
├── ensemble_weights.npy        # 필수
├── class_names.json            # 필수
└── webcam_local.py             # 실행 파일

2. 설정 조정
# 신뢰도 임계값 (낮추면 더 많이 인식)
CONFIDENCE_THRESHOLD = 0.7  # 0.5~0.9

# 예측 빈도 (낮추면 더 자주 예측, CPU 많이 사용)
FRAME_STRIDE = 10  # 5~15

# 중복 필터링 (같은 단어 재출력 대기 시간)
DUPLICATE_FILTER_SEC = 2.0  # 1~3초

키보드:
Q: 종료
R: 리셋 (버퍼 초기화)


3. 화면 설명
┌──────────────────────────────────────┐
│ Sign Language Recognition            │
│ Buffer: 30/30  |  FPS: 28.5          │
│                                      │
│ 안녕          92.3%                  │  ← 예측 결과
│                                      │
│     [웹캠 영상 + 손 랜드마크]         │
│                                      │
│ Recent:                              │
│ 안녕(92%) → 이름(88%) → 뭐(85%)      │
│                                      │
│ Press 'Q' to quit | 'R' to reset     │
└──────────────────────────────────────┘

4. 문제 해결
❌ "웹캠을 찾을 수 없습니다"
# 카메라 번호 변경
cap = cv2.VideoCapture(1)  # 0 → 1
❌ "모델을 찾을 수 없습니다"
# 파일 존재 확인
ls /Users/soyun/Desktop/s2t/26.02.02/*.keras
# 경로 확인
pwd

❌ FPS가 너무 낮음 (< 10)
# 해상도 낮추기
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 예측 빈도 낮추기
FRAME_STRIDE = 15
❌ 인식이 안 됨
# 1. 신뢰도 낮추기
CONFIDENCE_THRESHOLD = 0.5

# 2. MediaPipe 설정 낮추기
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
❌ 같은 단어만 계속 나옴
# 중복 필터링 시간 늘리기
DUPLICATE_FILTER_SEC = 3.0


### webcam_space.py
1. 화면 구성
┌────────────────────────────────────────────┐
│ Sign Language Recognition                  │
│ RECORDING               01:안녕 92%         │  ← 실시간 예측
│ Buffer: 30/30  |  FPS: 28.5                │
│                                            │
│        [웹캠 영상 + 랜드마크]               │
│                                            │
│ Words:                                     │
│ 01:안녕 → 11:이름 → 02:뭐                  │  ← 인식된 단어들
│                                            │
│ 문장: 안녕하세요! 이름이 뭐예요?           │  ← GPT 생성 문장
│ Space: Start/Stop & Generate | R: Reset    │
└────────────────────────────────────────────┘

2. 사용법
1단계: Space 누름
→ "RECORDING" 표시 (빨간색)
→ 실시간 인식 시작
2단계: 수어 동작
→ 오른쪽 상단에 실시간 예측 표시
→ 신뢰도 70% 이상이면 자동으로 단어 추가
→ 화면 하단에 쭉쭉 나열됨
3단계: Space 다시 누름
→ "GENERATING..." 표시
→ GPT가 자동으로 문장 생성
→ 맨 아래에 문장 표시
4단계: 다시 시작하려면
→ R 누름 (리셋)
→ Space 누름 (새로 시작)

3. 핵심 기능
- 실시간 연속 인식
    - 5프레임마다 자동 예측
    - 신뢰도 70% 이상만 추가
    - 1.5초 간격으로 중복 방지
- 자동 단어 추가
    - 새로운 단어면 자동 추가
    - 화면 하단에 실시간 표시
    - 화살표(→)로 구분
- 한 번에 문장 생성
    - Space 2번 누르면 즉시 GPT 호출
    - 생성 중 "GENERATING..." 표시

4. 설정
python# 신뢰도 임계값 (낮추면 더 많이 인식)
CONFIDENCE_THRESHOLD = 0.70  # 0.6~0.85

# 예측 빈도 (낮추면 더 자주 예측)
FRAME_STRIDE = 5  # 3~10

# 중복 방지 시간
# add_frame_and_predict 함수에서
current_time - self.last_prediction_time > 1.5  # 1~3초

5. 문제 해결
Q1: 단어가 너무 빨리 추가됨
# 중복 방지 시간 늘리기
current_time - self.last_prediction_time > 2.5  # 1.5 → 2.5
Q2: 단어가 안 추가됨
# 신뢰도 낮추기
CONFIDENCE_THRESHOLD = 0.60  # 0.70 → 0.60

# 예측 빈도 높이기
FRAME_STRIDE = 3  # 5 → 3
Q3: 같은 단어만 반복됨
# 중복 방지 시간 늘리기
current_time - self.last_prediction_time > 2.0  # 1.5 → 2.0