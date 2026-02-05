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
    if not kor_tokens:
        return ""

    # 0) 연속 중복 제거(원하면 삭제 가능)
    compact: List[str] = []
    for t in kor_tokens:
        if not compact or compact[-1] != t:
            compact.append(t)
    tokens = compact

    # 1) 분류
    times: List[str] = []
    places: List[str] = []
    persons: List[str] = []
    prons: List[str] = []
    nouns: List[str] = []
    verbs: List[str] = []
    extras: List[str] = []

    for t in tokens:
        typ = TOKEN_TYPE.get(t, "UNK")
        if typ == "TIME":
            times.append(t)
        elif typ in ("PLACE", "PLACEQ"):
            places.append(t)
        elif typ in ("PERSON", "PERSONQ"):
            persons.append(t)
        elif typ == "PRON":
            prons.append(t)
        elif typ == "NOUN":
            nouns.append(t)
        elif typ == "VERB":
            verbs.append(t)
        else:
            extras.append(t)

    # 2) verb 선택: 마지막 동사 우선(수어에서 동작이 끝에 오는 경우 많음)
    verb = verbs[-1] if verbs else None

    # 3) 관점 2: 주어=사람(PERSON) 우선, 목적어=나(PRON) 우선
    subj: Optional[str] = None
    obj: Optional[str] = None

    # 목적어는 '나'가 있으면 무조건 '나'
    if "나" in prons:
        obj = "나"

    # 주어는 PERSON 있으면 첫 번째
    if persons:
        subj = persons[0]
    else:
        # PERSON 없으면 '당신/우리/나' 중에서 '나'는 obj로 쓸 수 있으니 제외
        cand = [p for p in prons if p != obj]
        subj = cand[0] if cand else None

    # 목적어가 아직 없으면 남는 PERSON/명사 중 선택
    if obj is None:
        remain_persons = [p for p in persons if p != subj]
        if remain_persons:
            obj = remain_persons[0]
        elif nouns:
            obj = nouns[0]
        elif extras:
            obj = extras[0]

    # subj도 없고 obj만 나인 경우 -> "누군가" 보강
    if obj == "나" and subj is None:
        subj = "누군가"

    # 4) 문장 조립
    parts: List[str] = []

    if times:
        # "어제 저녁" 같이 2개까지만 붙임
        parts.append(" ".join(times[:2]))

    if places:
        # 기본 "~에서" (보다/먹다/만나다는 보통 '에서'가 자연스러움)
        place = places[0]
        if place == "어디":
            parts.append("어디에서")
        else:
            parts.append(_place_particle(place, prefer_from=True))

    if subj:
        parts.append(_as_subject(subj))

    if obj:
        # 목적어 '나'는 "나를"로
        if obj == "나":
            parts.append("나를")
        else:
            parts.append(_as_object(obj))

    if verb:
        parts.append(_to_past(verb))
    else:
        # 동사 없으면 최소 형태
        parts.append("했다")

    sent = " ".join(parts).strip()
    if sent and not sent.endswith("."):
        sent += "."
    return sent


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
