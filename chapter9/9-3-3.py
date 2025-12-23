# ================================
# 🇰🇷 한국어 불용어 제거 예제 (KoNLPy + 품사 기반)
# ================================

# 1️⃣ KoNLPy의 Okt 형태소 분석기 불러오기
# KoNLPy는 한국어 자연어처리를 위한 파이썬 패키지다.
# Okt(Open Korean Text)는 SNS나 일상 문장에 강한 분석기로,
# 명사, 동사, 형용사, 조사 등을 구분해준다.
from konlpy.tag import Okt

# 2️⃣ 형태소 분석기 객체 생성
okt = Okt()

# 3️⃣ 분석할 문장 예시
text = "이 정책은 결코 쉬운 결정이 아니지만, 국민의 안전을 위해 반드시 필요하다."

# 4️⃣ 형태소 분석 수행
# norm=True → '아니지만' 같은 문장을 기본형으로 정규화 ("아니지만" → "아니다")
# stem=True → 어간 추출, 예: "필요하다" → "필요"
pos = okt.pos(text, norm=True, stem=True)

# 형태소 분석 결과 예시:
# [('이', 'Determiner'), ('정책', 'Noun'), ('은', 'Josa'), ('결코', 'Adverb'),
#  ('쉽다', 'Adjective'), ('결정', 'Noun'), ('이', 'Josa'), ('아니다', 'Adjective'),
#  ('지만', 'Eomi'), ('국민', 'Noun'), ('의', 'Josa'), ('안전', 'Noun'),
#  ('을', 'Josa'), ('위해', 'PreEomi'), ('반드시', 'Adverb'), ('필요', 'Noun'),
#  ('하다', 'Verb'), ('.', 'Punctuation')]

# 5️⃣ 조사(Josa), 어미(Eomi), 구두점(Punctuation) 제거
# 조사(은, 는, 이, 가, 을, 를), 어미(다, 지만 등)는 문법적 역할만 하므로 제거 대상이다.
ban_pos = {'Josa', 'Eomi', 'Punctuation'}
filtered_by_pos = [w for w, p in pos if p not in ban_pos]

# 6️⃣ 도메인(분석 목적)에 맞게 커스텀 불용어 정의
# 예: 정책 분야 텍스트에서는 "정책", "위해" 같은 단어는 너무 빈번하므로 제거
custom_sw = {'것', '수', '정책', '위해'}

# 리스트 컴프리헨션을 사용하는 경우
# filtered = [w for w in filtered_by_pos if w not in custom_sw]

# for문을 사용하는 경우
filtered = []
for token in tokens:
    # 1) 알파벳인지 확인
    if not token.isalpha():
        continue

    # 2) 불용어인지 확인
    if token in stop_set:
        continue

    filtered.append(token)

# 7️⃣ 최종 결과 출력
print(filtered)

# 설명:
# - 형태소 분석을 통해 단어의 품사 정보를 얻는다.
# - 문법적 기능어(조사·어미)는 제거하고, 핵심 의미어(명사·형용사·부사 등)만 남긴다.
# - 불용어 목록은 과제에 맞게 조정하며, 반복 등장하지만 의미가 약한 단어를 제거한다.
# - 결과적으로 문장에서 ‘의미 중심 단어’만 남아, 분석 효율과 정확도가 향상된다.
