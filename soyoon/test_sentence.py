from translator import Translator, make_final_korean_sentence

# 테스트용 토큰들
sentence_kor = ["어제", "저녁", "서울", "부모님", "나", "보다"]
sentence_eng = ["yesterday", "evening", "seoul", "parents", "me", "see"]

translator = Translator()

en, ko = make_final_korean_sentence(
    translator,
    sentence_eng=sentence_eng,
    sentence_kor=sentence_kor
)

print("EN:", en)
print("KO:", ko)
