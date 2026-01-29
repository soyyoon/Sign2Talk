# Sign2Talk

한국어 수어 인식 기반 실시간 문장 변환 시스템

## 📌 소개

Sign2Talk은 웹캠을 통해 입력되는 수어(한국 수어) 영상을 컴퓨터 비전 기술로 실시간으로 인식하고, 이를 자연스러운 한국어 텍스트로 변환하는 프로토타입 인터페이스입니다. 이 프로젝트는 청각장애인과 비청각장애인 간의 실시간 양방향 의사소통을 지원하는 것을 목표로 합니다.

## 🔧 주요 기능

- 웹캠 기반 실시간 수어 동작 인식  
- 손동작 및 신체 자세 분석  
- 수어 인식 결과를 단어 시퀀스로 변환  
- OpenAI GPT-4o mini API를 활용한 자연스러운 한국어 문장 생성  
- 실시간 한국어 문장 출력  

## 📁 사용한 데이터셋

- 한국 수어(KSL) 동작 인식용 공개 비디오 데이터셋 활용

## 🧠 기술 스택

- Python  
- OpenCV, MediaPipe (수어 신호 전처리 및 특징 추출)  
- TensorFlow (수어 분류 모델)  
- OpenAI GPT-4o mini (문장 생성)  

## 🚀 설치 및 실행

아래는 로컬 환경에서 빠르게 실행해 보기 위한 예시입니다.

```bash
# 저장소 클론
git clone https://github.com/soyyoon/Sign2Talk.git
cd Sign2Talk

# 가상 환경 설정 (optional)
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 필요한 패키지 설치
pip install -r requirements.txt

# 환경 변수 설정
export OPENAI_API_KEY="your_api_key_here"
```

## 📂 폴더 구조
```text
Sign2Talk/
├── main.py                   # 실시간 수어 인식 및 문장 생성 메인 실행 파일
├── recognizer.py             # 수어 인식 로직 (시계열 처리)
├── recognizer_v2.py          # 개선된 수어 인식 모델 로직
├── features.py               # 손/포즈 기반 특징 추출
├── preprocess.py             # 입력 데이터 전처리
├── models.py                 # 모델 구성 및 로드 함수
├── labels.py                 # 수어 라벨 정의
├── class_names.json          # 클래스 인덱스 정보
├── mean.npy                  # 정규화 평균 값
├── std.npy                   # 정규화 표준편차 값
├── slim_v1.keras             # 수어 인식 모델 (앙상블)
├── slim_v2.keras
├── slim_v3.keras
├── config.py                 # 환경 및 하이퍼파라미터 설정
├── ui.py                     # 사용자 인터페이스 관련 코드
├── translator.py             # 수어 인식 결과 후처리 및 문장 구성
├── requirements.txt          # 프로젝트 의존성 목록
└── README.md                 # 프로젝트 설명
```

## 🧩 확장 가능성

Sign2Talk은 다음과 같은 기능 확장이 가능합니다:

- 감정 표현(폰트 스타일, 이모지, TTS 등)을 포함한 풍부한 출력  
- 다양한 사용자 환경을 반영한 데이터셋 확장  
- 양방향 소통 기능 강화  
