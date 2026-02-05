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
            "You are a Korean language corrector that refines Korean Sign Language (KSL) recognition results into sentences.\n"
            "The input is a sequence of meaningful words that may have broken grammar.\n"
            "You must NEVER add information that is not present in the input."
        )

        user_prompt = f"""
Input word list:
{kor_tokens}

Rules:
- Preserve the meaning of input words as much as possible
- NEVER imagine or add actions, times, objects, emotions, or any other information which is not in the input
- Only minimally correct particles (postpositions) and word order
- Use casual speech (반말), NOT formal/polite speech (존댓말)
- Output as a single natural declarative sentence
- Provide ONLY the sentence - no explanations, commentary, or quotation marks
- When uncertain, be conservative and keep it short
- Result MUST only be in KOREAN

Forbidden Additions:
- No intensifiers (very, really, extremely) unless in input
- No connecting words (however, therefore, but) between single sentences
- No pronouns (I, you, he/she) unless explicitly signed
- No descriptive adjectives beyond what's in the input
- No temporal context (today, yesterday, later) unless present

Output Format:
- Maximum one sentence per input
- Prefer shorter constructions when ambiguous
- If input is fragmented/unclear, output the most literal interpretation
- Remove redundant particles rather than adding missing ones

CRITICAL - Priority 1:
1. NEVER infer implied subjects (I/you/we) from context
2. NEVER add causality (because, so, therefore)
3. NEVER specify unmentioned locations/times
4. NEVER elaborate on emotions/states
5. When multiple interpretations exist → choose the most literal/conservative

Priority 2 - Minimal Correction:
1. Only add particles that are 100% necessary for basic grammar
2. Prefer dropping particles over guessing wrong ones
3. Keep original word order unless grammatically impossible
4. Don't "fix" regional dialects or colloquialisms

Example Inputs and Outputs:
- Input: ["나", "밥", "먹다"]
  Output: "나는 밥을 먹는다."
- Input: ["어제", "학교", "가다"]
  Output: "어제 학교에 갔다."
- Input: ["나", "보다", "영화"]
  Output: "나는 영화를 본다."
- Input: ["친구", "나", "버스", "타다"]
  Output: "친구와 나는 버스를 탄다."
- Input: ["서울", "부모님", "보다"]
  Output: "서울에서 부모님을 본다."
- Input: ["여동생", "어제", "영화", "보다"]
  Output: "여동생은 어제 영화를 봤다."
- Input: ["우리", "지금", "지하철", "타다"]
  Output: "우리는 지금 지하철을 탄다."
- Input: ["서울", "어디", "위치", "묻다"]
  Output: "서울이 어디에 위치했는지 묻는다."
- Input: ["가족", "저녁", "음식", "먹다"]
  Output: "가족은 저녁에 음식을 먹는다."
- Input: ["어제", "시험", "끝", "괜찮다"]
  Output: "어제 시험이 끝나서 괜찮다."
- Input: ["지금", "버스", "타다", "도착하다"]
  Output: "지금 버스를 타고 도착한다."
- Input: ["가족", "나이", "몇", "묻다"]
  Output: "가족의 나이가 몇인지 묻는다."
- Input: ["저녁", "비빔밥", "먹다", "좋다"]
  Output: "저녁에 비빔밥을 먹어서 좋다."
- Input: ["우리", "노력", "마침내", "성공"]
  Output: "우리는 노력해서 마침내 성공했다."

Output:
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
