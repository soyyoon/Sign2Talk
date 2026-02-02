Sign2Talk (S2T)
수어 동작을 실시간으로 인식하여 텍스트 및 자연스러운 문장으로 변환하는 프로젝트

1. Data Preprocessing (data_preprocessing.py)
학습 품질을 높이기 위해 데이터의 속도, 체격, 품질을 모두 정규화함.

속도 정규화: 사람마다 다른 수어 속도를 실제 동작 구간 추출 후 30 프레임으로 리샘플링하여 동일하게 인식함.

공간 정규화: 어깨 너비를 기준으로 모든 거리 feature를 나누어 체격 차이를 보정하고 동작 자체만 학습함.

품질 검증: MediaPipe 랜드마크 감지 여부를 확인하여 유효 프레임 비율, 데이터 밀도, 이상치 유무 등 3가지 검증을 통과한 데이터만 저장함.

보간 및 이상치 제거: 일시적 감지 실패는 전후 프레임 평균으로 채우고, Z-score 기반으로 극단적 오류값을 제거함.

미러링 증강: 데이터를 2배로 증강하여 좌우 방향에 상관없는 불변성을 확보함.

2. Model Training (training_model.py)
총 67개 클래스, 2,420개 샘플을 활용해 4가지 아키텍처를 학습시킨 후 앙상블함.

모델 아키텍처

V1: Multi-scale Temporal CNN: 다양한 dilation을 적용해 여러 속도 패턴을 학습하며 속도 변화에 가장 강건함.

V2: CNN + Attention: Global Average/Max Pooling을 병렬로 사용해 중요한 특징을 강조함.

V3: Transformer: Self-attention을 통해 수어 시퀀스의 관계와 장거리 의존성을 포착함.

V4: Hybrid (CNN + Transformer): CNN으로 지역 패턴을 추출하고 Transformer로 시퀀스 관계를 포착하는 하이브리드 방식.

학습 전략

손실 함수: Focal Loss와 Label Smoothing을 적용해 클래스 불균형을 해소하고 과적합을 방지함.

데이터 증강: 미세 노이즈(±1%) 및 스케일링(0.98~1.02배) 증강을 적용함.

최적화: AdamW 옵티마이저와 ReduceLROnPlateau 스케줄러를 사용해 학습 효율을 높임.

가중 앙상블: Grid Search로 최적 가중치를 찾아 4개 모델의 예측을 결합함.

3. Real-time Inference (webcam.py & webcam_space.py)
단어 단위 인식 (webcam.py)

기능: 실시간으로 단일 수어 단어를 예측하고 신뢰도를 표시함.

조작: Q(종료), R(리셋).

주요 설정: CONFIDENCE_THRESHOLD(신뢰도 임계값), FRAME_STRIDE(예측 빈도) 조정 가능.

문장 완성 기능 (webcam_space.py)

기능: 연속된 수어 단어를 인식한 뒤 GPT를 통해 자연스러운 문장으로 변환함.

사용법:

Space 누름: 녹화 시작 (RECORDING 표시).

수어 동작: 신뢰도 기준 충족 시 자동으로 단어 리스트에 추가.

Space 다시 누름: GPT 호출 및 최종 문장 생성.

R 누름: 버퍼 초기화 및 리셋.

4. Troubleshooting
카메라 미인식: cv2.VideoCapture(0)에서 인덱스 번호를 1 등으로 변경.

인식률 저하: CONFIDENCE_THRESHOLD를 0.5~0.6 정도로 낮추거나 조명 및 배경 확인.

속도 문제: FRAME_STRIDE 값을 높여 연산 부하를 줄이거나 해상도 조정.