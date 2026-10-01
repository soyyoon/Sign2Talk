# sign2talk/translator.py
import re
from typing import List, Dict, Optional, Tuple

# ??踰덉뿭湲??좎튂 import???좎??섎릺, ?ㅼ젣濡쒕뒗 ?ъ슜 ???섎룄濡??덉쟾 泥섎━
# (?섍꼍??torch/transformers ?놁쑝硫?recognizer媛 二쎌뼱??try濡?媛먯뙃?덈떎)
try:
    import torch  # type: ignore
    from transformers import MarianMTModel, MarianTokenizer  # type: ignore
except Exception:
    torch = None
    MarianMTModel = None
    MarianTokenizer = None

import os

def ko_sentence_from_gpt(kor_tokens: List[str]) -> str:
    """
    kor_tokens(?? ['?댁젣','???,'?쒖슱','遺紐⑤떂','??,'蹂대떎']) -> ?먯뿰?ㅻ윭???쒓뎅??臾몄옣 ?앹꽦
    ?섍꼍蹂??OPENAI_API_KEY ?꾩슂
    """
    if not kor_tokens:
        return ""

    api_key = os.getenv("YOUR_OPENAI_API_KEY")
    if not api_key:
        # ???놁쑝硫?洹몃깷 鍮?臾몄옄??諛섑솚 ??fallback ?寃???
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"""
?덈뒗 ?쒓뎅??臾몄옣 ?앹꽦湲곗빞.
?낅젰? ?섏뼱 ?몄떇?쇰줈 ?섏삩 '?좏겙 ?섏뿴'?닿퀬, 異쒕젰? ?먯뿰?ㅻ윭???쒓뎅????臾몄옣?댁빞.

洹쒖튃:
- ?낅젰 ?좏겙???섎?瑜?理쒕???蹂댁〈??
- 遺덊븘?뷀븳 ?곸긽/異붽? ?뺣낫 湲덉?.
- ??臾몄옣?쇰줈, 議대뙎留?留먭퀬 ?됱꽌臾??덈떎/?덈떎.) ?ㅽ???
- ?μ냼媛 ?덉쑝硫?蹂댄넻 "~?먯꽌", ?쒓컙???덉쑝硫?臾몄옣 ?욎뿉.
- 異쒕젰? 臾몄옣留? ?곗샂???ㅻ챸/踰덊샇 ?놁씠.

?낅젰 ?좏겙: {kor_tokens}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You convert token lists into a single natural Korean sentence."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        # 留덉묠??蹂댁젙
        if text and text[-1] not in ".!?":
            text += "."
        return text

    except Exception:
        # API ?먮윭/紐⑤뱢 ?먮윭 ?깆? fallback
        return ""

# =========================
# (?듭뀡) 踰덉뿭 紐⑤뜽 ?대쫫 (?꾩옱???ъ슜 ????
# =========================
MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-ko"


class EnKoTranslator:
    """
    ???꾩옱 ?꾨줈?앺듃?먯꽌??'?덈? 踰덉뿭湲곕? ?곗? ?딅뒗??媛 紐⑺몴?쇱꽌
    - torch/transformers媛 ?놁뼱???숈옉?섎룄濡??덉쟾?섍쾶 留뚮뱾?덇퀬
    - translate()瑜??몄텧?대룄 ?낅젰??洹몃?濡?諛섑솚?섎룄濡?泥섎━?덉뒿?덈떎.
    """
    def __init__(self):
        self.enabled = (MarianTokenizer is not None and MarianMTModel is not None and torch is not None)
        self.tokenizer = None
        self.model = None

        # 踰덉뿭湲?鍮꾪솢??湲곕낯): ?몃? 紐⑤뜽 ?ㅼ슫濡쒕뱶/?섏〈???ㅻ쪟 諛⑹?
        self.enabled = False

        # 留뚯빟 ?섏쨷???뺣쭚 ?곌퀬 ?띠쑝硫?enabled=True濡?諛붽씀怨??꾨옒 二쇱꽍 ?댁젣
        # if self.enabled:
        #     self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        #     self.model = MarianMTModel.from_pretrained(MODEL_NAME)
        #     self.model.eval()

    def translate(self, en_text: str) -> str:
        # ???덈? ?ъ슜 ????(?덉쟾?섍쾶 洹몃?濡?諛섑솚)
        return en_text.strip()


def gloss_to_simple_english(gloss_words: List[str]) -> str:
    """
    gloss(?⑥뼱 ?섏뿴)??'蹂닿린 醫뗭? ?곸뼱 ?붾쾭洹?臾몄옣'?쇰줈留?留뚮뱶??理쒖냼 洹쒖튃.
    (踰덉뿭 ?덉쭏 紐⑹쟻 X, ?붾㈃/濡쒓렇 ?붾쾭洹?紐⑹쟻)
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
    ?????댁긽 '?섎せ??臾몄옣?낅땲??濡?留됱? ?딅뒗 諛⑺뼢.
    (?④꺼?먭릿 ?섏?留??꾩옱 ?뚯씠?꾨씪?몄뿉??媛뺤젣 李⑤떒?섏? ?딅룄濡??ъ슜)
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
# 1) ?쒓뎅???뺥깭 ?좏떥
# =========================
def _has_batchim(korean_word: str) -> bool:
    """留덉?留?湲??諛쏆묠 ?щ?"""
    if not korean_word:
        return False
    ch = korean_word[-1]
    code = ord(ch)
    if code < 0xAC00 or code > 0xD7A3:
        return False
    return ((code - 0xAC00) % 28) != 0

def _josa(word: str, pair: Tuple[str, str]) -> str:
    """(諛쏆묠?덉쓬, 諛쏆묠?놁쓬) 以??좏깮"""
    return pair[0] if _has_batchim(word) else pair[1]

def _as_subject(noun: str) -> str:
    """
    愿??2(?곹솴 ?ㅻ챸泥??먯꽌??
    - '?????꾩슂?섎㈃ '?닿?'濡??곗씪 ???덉?留?
    - 湲곕낯? ?몃? 二쇱뼱(PERSON)媛 ?덉쑝硫?洹멸구 二쇱뼱濡??곕뒗 寃?紐⑹쟻.
    """
    return noun + _josa(noun, ("??, "媛"))

def _as_topic(noun: str) -> str:
    return noun + _josa(noun, ("?", "??))

def _as_object(noun: str) -> str:
    return noun + _josa(noun, ("??, "瑜?))

def _place_particle(place: str, prefer_from: bool = True) -> str:
    # 湲곕낯? "~?먯꽌"
    return place + ("?먯꽌" if prefer_from else "??)


# =========================
# 2) ?숈궗 怨쇨굅??(?꾩＜ 理쒖냼)
# =========================
PAST_VERB: Dict[str, str] = {
    "蹂대떎": "遊ㅻ떎",
    "癒밸떎": "癒뱀뿀??,
    "留뚮굹??: "留뚮궗??,
    "?쎈떎": "?쎌뿀??,
    "嫄룸떎": "嫄몄뿀??,
    "媛瑜댁튂??: "媛瑜댁낀??,
    "諛쏅떎": "諛쏆븯??,
    "?꾩갑": "?꾩갑?덈떎",
    "?묒듅": "?묒듅?덈떎",
    "?뚭컻": "?뚭컻?덈떎",
}

def _to_past(verb: str) -> str:
    return PAST_VERB.get(verb, verb if verb.endswith("??) else verb + "?덈떎")


# =========================
# 3) ?좏겙 遺꾨쪟(?뺤옣 ?ъ씤??
#    - ?⑥뼱媛 ?섏뼱?섎㈃ ?ш린留?議곌툑??異붽??섎㈃ ??
# =========================
TOKEN_TYPE: Dict[str, str] = {
    # TIME
    "?댁젣": "TIME", "?ㅻ뒛": "TIME", "?댁씪": "TIME",
    "???: "TIME", "?꾩묠": "TIME", "?먯떖": "TIME", "吏湲?: "TIME",
    "10遺?: "TIME", "?쒓컙": "TIME",

    # PLACE / PLACEQ
    "?쒖슱": "PLACE", "吏?섏쿋": "PLACE", "踰꾩뒪": "PLACE", "怨?: "PLACE",
    "?대뵒": "PLACEQ",

    # PERSON / PERSONQ
    "遺紐⑤떂": "PERSON", "媛議?: "PERSON", "?щ룞??: "PERSON", "?꾧뎄": "PERSONQ",

    # PRON (?꾩슂?섎㈃ ?ъ슜)
    "??: "PRON", "?뱀떊": "PRON", "?곕━": "PRON",

    # VERB
    "蹂대떎": "VERB", "癒밸떎": "VERB", "留뚮굹??: "VERB", "?쎈떎": "VERB",
    "嫄룸떎": "VERB", "媛瑜댁튂??: "VERB", "諛쏅떎": "VERB", "?꾩갑": "VERB",
    "?묒듅": "VERB", "?뚭컻": "VERB",

    # NOUN (OBJECT ?꾨낫)
    "鍮꾨퉼諛?: "NOUN", "?뚯떇": "NOUN", "?대??꾪솕": "NOUN", "?곹솕": "NOUN",
    "?대쫫": "NOUN", "踰덊샇": "NOUN", "?섏씠": "NOUN", "痍⑤?": "NOUN",
}


# =========================
# 4) ?듭떖: 愿??2(?곹솴 ?ㅻ챸泥? 臾몄옣 留뚮뱾湲?
#    紐⑺몴: "?댁젣 ????쒖슱 遺紐⑤떂 ??蹂대떎"
#        -> "?댁젣 ????쒖슱?먯꽌 遺紐⑤떂???섎? 遊ㅻ떎."
# =========================
def ko_sentence_from_kor_tokens(kor_tokens: List[str]) -> str:
    if not kor_tokens:
        return ""

    # 0) ?곗냽 以묐났 ?쒓굅(?먰븯硫???젣 媛??
    compact: List[str] = []
    for t in kor_tokens:
        if not compact or compact[-1] != t:
            compact.append(t)
    tokens = compact

    # 1) 遺꾨쪟
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

    # 2) verb ?좏깮: 留덉?留??숈궗 ?곗꽑(?섏뼱?먯꽌 ?숈옉???앹뿉 ?ㅻ뒗 寃쎌슦 留롮쓬)
    verb = verbs[-1] if verbs else None

    # 3) 愿??2: 二쇱뼱=?щ엺(PERSON) ?곗꽑, 紐⑹쟻????PRON) ?곗꽑
    subj: Optional[str] = None
    obj: Optional[str] = None

    # 紐⑹쟻?대뒗 '??媛 ?덉쑝硫?臾댁“嫄?'??
    if "?? in prons:
        obj = "??

    # 二쇱뼱??PERSON ?덉쑝硫?泥?踰덉㎏
    if persons:
        subj = persons[0]
    else:
        # PERSON ?놁쑝硫?'?뱀떊/?곕━/?? 以묒뿉??'????obj濡??????덉쑝???쒖쇅
        cand = [p for p in prons if p != obj]
        subj = cand[0] if cand else None

    # 紐⑹쟻?닿? ?꾩쭅 ?놁쑝硫??⑤뒗 PERSON/紐낆궗 以??좏깮
    if obj is None:
        remain_persons = [p for p in persons if p != subj]
        if remain_persons:
            obj = remain_persons[0]
        elif nouns:
            obj = nouns[0]
        elif extras:
            obj = extras[0]

    # subj???녾퀬 obj留??섏씤 寃쎌슦 -> "?꾧뎔媛" 蹂닿컯
    if obj == "?? and subj is None:
        subj = "?꾧뎔媛"

    # 4) 臾몄옣 議곕┰
    parts: List[str] = []

    if times:
        # "?댁젣 ??? 媛숈씠 2媛쒓퉴吏留?遺숈엫
        parts.append(" ".join(times[:2]))

    if places:
        # 湲곕낯 "~?먯꽌" (蹂대떎/癒밸떎/留뚮굹?ㅻ뒗 蹂댄넻 '?먯꽌'媛 ?먯뿰?ㅻ윭?)
        place = places[0]
        if place == "?대뵒":
            parts.append("?대뵒?먯꽌")
        else:
            parts.append(_place_particle(place, prefer_from=True))

    if subj:
        parts.append(_as_subject(subj))

    if obj:
        # 紐⑹쟻??'????"?섎?"濡?
        if obj == "??:
            parts.append("?섎?")
        else:
            parts.append(_as_object(obj))

    if verb:
        parts.append(_to_past(verb))
    else:
        # ?숈궗 ?놁쑝硫?理쒖냼 ?뺥깭
        parts.append("?덈떎")

    sent = " ".join(parts).strip()
    if sent and not sent.endswith("."):
        sent += "."
    return sent


def make_final_korean_sentence(translator, sentence_eng: List[str], sentence_kor: List[str]):
    en_sent = " ".join(sentence_eng).strip()
    if en_sent:
        en_sent = en_sent[0].upper() + en_sent[1:] + "."

    # 1) GPT ?곗꽑
    ko_sent = ko_sentence_from_gpt(sentence_kor)

    # 2) GPT ?ㅽ뙣?섎㈃ 湲곗〈 洹쒖튃 湲곕컲 fallback
    if not ko_sent.strip():
        ko_sent = ko_sentence_from_kor_tokens(sentence_kor)

    # 3) 洹몃옒??鍮꾨㈃ ?좏겙 ?섏뿴
    if not ko_sent.strip():
        ko_sent = " ".join(sentence_kor) if sentence_kor else "?몄떇???⑥뼱媛 ?놁뒿?덈떎."

    return en_sent, ko_sent
