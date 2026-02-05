import os
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()

# =========================================================
# Translator (기존 구조 유지 – 실제 API 번역은 사용 안 함)
# =========================================================

class Translator:
    def __init__(self):
        # 기존 코드 호환용
        self.enabled = False

    def translate(self, en_text: str) -> str:
        # ❗ 의도적으로 아무 것도 하지 않음 (기존 설계 유지)
        return en_text.strip()


# =========================================================
# GPT-4o 기반 문장 보정기 (추가된 부분)
# =========================================================

def ko_sentence_from_gpt(kor_tokens: List[str]) -> str:
    """
    GPT-4o를 사용해 '단어 나열 → 자연스러운 한국어 문장'으로 보정
    ❗ 정보 추가 / 상상 절대 금지
    """
    if not kor_tokens:
        return ""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system_prompt = (
            "너는 수어 인식 결과를 문장으로 다듬는 한국어 보정기이다.\n"
            "입력은 의미 단어들의 나열이며 문법이 깨져 있을 수 있다.\n"
            "절대로 입력에 없는 정보를 추가하지 마라."
        )

        user_prompt = f"""
입력 단어 목록:
{kor_tokens}

규칙:
- 입력 단어의 의미를 최대한 그대로 유지할 것
- 없는 동작, 시간, 대상, 감정 등을 상상해서 추가하지 말 것
- 조사, 어순만 최소한으로 보정
- 존댓말 사용 금지
- 하나의 자연스러운 평서문으로 출력
- 설명, 해설, 따옴표 없이 문장만 출력
- 불확실하면 보수적으로 짧게 구성

출력:
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
        )

        text = response.choices[0].message.content.strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text

    except Exception as e:
        print("GPT sentence error:", e)
        return ""


# =========================================================
# 기존 규칙 기반 문장 생성 (❗ 수정하지 않음)
# =========================================================

TOKEN_TYPE = {
    "나": "subject",
    "너": "subject",
    "우리": "subject",
    "부모님": "subject",

    "밥": "object",
    "학교": "object",
    "서울": "place",

    "가다": "verb",
    "오다": "verb",
    "보다": "verb",
    "먹다": "verb",

    "어제": "time",
    "오늘": "time",
    "내일": "time",
}

def ko_sentence_from_kor_tokens(kor_tokens: List[str]) -> str:
    if not kor_tokens:
        return ""

    time = []
    subject = []
    obj = []
    verb = []
    etc = []

    for tok in kor_tokens:
        t = TOKEN_TYPE.get(tok)
        if t == "time":
            time.append(tok)
        elif t == "subject":
            subject.append(tok)
        elif t == "object" or t == "place":
            obj.append(tok)
        elif t == "verb":
            verb.append(tok)
        else:
            etc.append(tok)

    parts = []
    parts.extend(time)
    parts.extend(subject)
    parts.extend(obj)
    parts.extend(etc)
    parts.extend(verb)

    if not parts:
        return ""

    sent = " ".join(parts)
    return sent + "."


# =========================================================
# 최종 문장 생성 진입점 (기존 구조 유지 + GPT 우선)
# =========================================================

def make_final_korean_sentence(
    translator: Translator,
    sentence_eng: List[str],
    sentence_kor: List[str]
) -> Tuple[str, str]:

    # 영어 디버그 문장 (기존 용도 유지)
    en_sent = " ".join(sentence_eng).strip()
    if en_sent:
        en_sent = en_sent[0].upper() + en_sent[1:] + "."

    # ✅ 1️⃣ GPT-4o 우선
    ko_sent = ko_sentence_from_gpt(sentence_kor)

    # ✅ 2️⃣ GPT 실패 → 기존 규칙 기반
    if not ko_sent.strip():
        ko_sent = ko_sentence_from_kor_tokens(sentence_kor)

    # ✅ 3️⃣ 최후 fallback
    if not ko_sent.strip():
        ko_sent = " ".join(sentence_kor) if sentence_kor else "인식된 단어가 없습니다."

    return en_sent, ko_sent
