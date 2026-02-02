# sign2talk/translator.py
import re
from typing import List, Dict, Optional, Tuple

# ✅ 번역기/토치 import는 유지하되, 실제로는 사용 안 하도록 안전 처리
# (환경에 torch/transformers 없으면 recognizer가 죽어서 try로 감쌉니다)
try:
    import torch  # type: ignore
    from transformers import MarianMTModel, MarianTokenizer  # type: ignore
except Exception:
    torch = None
    MarianMTModel = None
    MarianTokenizer = None


# =========================
# (옵션) 번역 모델 이름 (현재는 사용 안 함)
# =========================
MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-ko"


class EnKoTranslator:
    """
    ✅ 현재 프로젝트에서는 '절대 번역기를 쓰지 않는다'가 목표라서
    - torch/transformers가 없어도 동작하도록 안전하게 만들었고
    - translate()를 호출해도 입력을 그대로 반환하도록 처리했습니다.
    """
    def __init__(self):
        self.enabled = (MarianTokenizer is not None and MarianMTModel is not None and torch is not None)
        self.tokenizer = None
        self.model = None

        # 번역기 비활성(기본): 외부 모델 다운로드/의존성/오류 방지
        self.enabled = False

        # 만약 나중에 정말 쓰고 싶으면 enabled=True로 바꾸고 아래 주석 해제
        # if self.enabled:
        #     self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        #     self.model = MarianMTModel.from_pretrained(MODEL_NAME)
        #     self.model.eval()

    def translate(self, en_text: str) -> str:
        # ✅ 절대 사용 안 함 (안전하게 그대로 반환)
        return en_text.strip()


def gloss_to_simple_english(gloss_words: List[str]) -> str:
    """
    gloss(단어 나열)을 '보기 좋은 영어 디버그 문장'으로만 만드는 최소 규칙.
    (번역 품질 목적 X, 화면/로그 디버그 목적)
    """
    if not gloss_words:
        return ""

    w = [x.replace("_", " ") for x in gloss_words]

    mapping = {
        "me": "I",
        "hi": "Hello",
        "glad": "Nice",
        "thank": "Thank you",
        "sorry": "Sorry",
        "how many": "How many",
        "cell phone": "cellphone",
        "ten minutes": "ten minutes",
        "one hour": "one hour",
    }
    w2 = [mapping.get(token, token) for token in w]

    q_words = {"what", "where", "when", "who", "how many"}
    is_question = any(t.lower() in q_words for t in w2)

    sent = " ".join(w2).strip()
    if not sent:
        return ""

    sent = sent[0].upper() + sent[1:] if len(sent) >= 2 else sent.upper()
    sent += "?" if is_question else "."
    return sent


def is_bad_sentence(ko_text: str) -> bool:
    """
    ✅ 더 이상 '잘못된 문장입니다'로 막지 않는 방향.
    (남겨두긴 하지만 현재 파이프라인에서 강제 차단하지 않도록 사용)
    """
    if ko_text is None:
        return True
    t = ko_text.strip()
    if len(t) < 2:
        return True
    if re.fullmatch(r"[A-Za-z0-9\s\.\?\!_/-]+", t) is not None:
        return True
    if "UNK" in t or "unknown" in t.lower():
        return True
    return False


# =========================
# 1) 한국어 형태 유틸
# =========================
def _has_batchim(korean_word: str) -> bool:
    """마지막 글자 받침 여부"""
    if not korean_word:
        return False
    ch = korean_word[-1]
    code = ord(ch)
    if code < 0xAC00 or code > 0xD7A3:
        return False
    return ((code - 0xAC00) % 28) != 0

def _josa(word: str, pair: Tuple[str, str]) -> str:
    """(받침있음, 받침없음) 중 선택"""
    return pair[0] if _has_batchim(word) else pair[1]

def _as_subject(noun: str) -> str:
    """
    관점 2(상황 설명체)에서는
    - '나'도 필요하면 '내가'로 쓰일 수 있지만,
    - 기본은 외부 주어(PERSON)가 있으면 그걸 주어로 쓰는 게 목적.
    """
    return noun + _josa(noun, ("이", "가"))

def _as_topic(noun: str) -> str:
    return noun + _josa(noun, ("은", "는"))

def _as_object(noun: str) -> str:
    return noun + _josa(noun, ("을", "를"))

def _place_particle(place: str, prefer_from: bool = True) -> str:
    # 기본은 "~에서"
    return place + ("에서" if prefer_from else "에")


# =========================
# 2) 동사 과거형 (아주 최소)
# =========================
PAST_VERB: Dict[str, str] = {
    "보다": "봤다",
    "먹다": "먹었다",
    "만나다": "만났다",
    "읽다": "읽었다",
    "걷다": "걸었다",
    "가르치다": "가르쳤다",
    "받다": "받았다",
    "도착": "도착했다",
    "탑승": "탑승했다",
    "소개": "소개했다",
}

def _to_past(verb: str) -> str:
    return PAST_VERB.get(verb, verb if verb.endswith("다") else verb + "했다")


# =========================
# 3) 토큰 분류(확장 포인트)
#    - 단어가 늘어나면 여기만 조금씩 추가하면 됨
# =========================
TOKEN_TYPE: Dict[str, str] = {
    # TIME
    "어제": "TIME", "오늘": "TIME", "내일": "TIME",
    "저녁": "TIME", "아침": "TIME", "점심": "TIME", "지금": "TIME",
    "10분": "TIME", "시간": "TIME",

    # PLACE / PLACEQ
    "서울": "PLACE", "지하철": "PLACE", "버스": "PLACE", "곳": "PLACE",
    "어디": "PLACEQ",

    # PERSON / PERSONQ
    "부모님": "PERSON", "가족": "PERSON", "여동생": "PERSON", "누구": "PERSONQ",

    # PRON (필요하면 사용)
    "나": "PRON", "당신": "PRON", "우리": "PRON",

    # VERB
    "보다": "VERB", "먹다": "VERB", "만나다": "VERB", "읽다": "VERB",
    "걷다": "VERB", "가르치다": "VERB", "받다": "VERB", "도착": "VERB",
    "탑승": "VERB", "소개": "VERB",

    # NOUN (OBJECT 후보)
    "비빔밥": "NOUN", "음식": "NOUN", "휴대전화": "NOUN", "영화": "NOUN",
    "이름": "NOUN", "번호": "NOUN", "나이": "NOUN", "취미": "NOUN",
}


# =========================
# 4) 핵심: 관점 2(상황 설명체) 문장 만들기
#    목표: "어제 저녁 서울 부모님 나 보다"
#        -> "어제 저녁 서울에서 부모님이 나를 봤다."
# =========================
def ko_sentence_from_kor_tokens(kor_tokens: List[str]) -> str:
    if not kor_tokens: return ""

    # 1. 분류 (TOKEN_TYPE 기반으로 그룹화)
    cats = {"TIME":[], "PLACE":[], "PERSON":[], "PRON":[], "NOUN":[], "VERB":[], "EXTRA":[]}
    for t in kor_tokens:
        typ = TOKEN_TYPE.get(t, "EXTRA")
        if typ in cats: cats[typ].append(t)
        else: cats["EXTRA"].append(t)

    subj, obj = None, None
    
    # 2. 역할 결정 전략 (일반화된 우선순위)
    
    # 전략 A: '사람'이나 '대명사'가 있으면 우선적으로 주어 후보로 올림
    human_candidates = cats["PRON"] + cats["PERSON"]
    
    if human_candidates:
        subj = human_candidates[0] # 첫 번째 사람/대명사를 주어로 선택
        
        # 주어를 제외하고 남은 사람이나 사물(NOUN)이 있으면 목적어로 설정
        remaining_objects = [h for h in human_candidates if h != subj] + cats["NOUN"]
        if remaining_objects:
            obj = remaining_objects[0]
    
    # 전략 B: 사람/대명사가 없고 사물(NOUN)만 있을 때 (예: "버스", "도착하다")
    elif cats["NOUN"]:
        subj = cats["NOUN"][0]
        if len(cats["NOUN"]) > 1:
            obj = cats["NOUN"][1]

    # 3. 조사 유틸리티를 통한 문장 조립
    parts = []
    if cats["TIME"]: parts.append(" ".join(cats["TIME"][:2]))
    if cats["PLACE"]: parts.append(_place_particle(cats["PLACE"][0]))
    
    if subj: parts.append(_as_subject(subj)) # '나' -> '내가', '버스' -> '버스가'
    if obj: parts.append(_as_object(obj))   # '나' -> '나를', '영화' -> '영화를'
    
    verb = cats["VERB"][-1] if cats["VERB"] else None
    parts.append(_to_past(verb) if verb else "했다")

    sent = " ".join(parts).strip()
    return sent + "." if sent else ""


def make_final_korean_sentence(translator, sentence_eng: List[str], sentence_kor: List[str]):
    """
    - sentence_eng: 영어 라벨 토큰 리스트 (디버그용)
    - sentence_kor: 한국어 라벨 토큰 리스트 (문장화에 사용)
    ✅ 번역기 절대 사용 안 함 (API/모델 다운로드 비용/오류 방지)
    """
    en_sent = " ".join(sentence_eng).strip()
    if en_sent:
        en_sent = en_sent[0].upper() + en_sent[1:] + "."

    ko_sent = ko_sentence_from_kor_tokens(sentence_kor)

    if not ko_sent.strip():
        ko_sent = " ".join(sentence_kor) if sentence_kor else "인식된 단어가 없습니다."

    return en_sent, ko_sent