# Sign2Talk

한국어 수어 인식 기반 실시간 음성 변환 시스템

## 📌 소개

Sign2Talk은 웹캠을 통해 입력되는 수어(한국 수어) 영상을 컴퓨터 비전 기술로 실시간으로 인식하고, 이를 자연스러운 한국어 텍스트 및 음성으로 변환하는 프로토타입 인터페이스입니다. 이 프로젝트는 청각장애인과 비청각장애인 간의 실시간 양방향 의사소통을 지원하는 것을 목표로 합니다.

## 🔧 주요 기능

- 웹캠 기반 실시간 수어 동작 인식  
- 손동작 및 신체 자세 분석  
- 수어 인식 결과를 단어 시퀀스로 변환  
- OpenAI GPT-4o mini API를 활용한 자연스러운 한국어 문장 생성  
- 실시간 한국어 텍스트 및 음성 출력  

## 📁 사용한 데이터셋

- 한국 수어(KSL) 동작 인식용 공개 비디오 데이터셋 활용

## 🧠 기술 스택

- Python  
- OpenCV, MediaPipe (수어 신호 전처리 및 특징 추출)  
- TensorFlow (수어 분류 모델)  
- OpenAI GPT-4o mini (문장 생성)  
- Streamlit / FastAPI (데모 인터페이스, 선택적)  

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
