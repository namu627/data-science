# ================================
# 🧩 영어 불용어 제거 예제
# ================================

# 1️⃣ NLTK 라이브러리 불러오기
# NLTK(Natural Language Toolkit)는 자연어처리용 파이썬 대표 라이브러리다.
# 단어 토큰화, 불용어 제거, 형태 분석 등에 자주 사용된다.
import nltk
from nltk.corpus import stopwords

# 2️⃣ NLTK의 'stopwords' 데이터 다운로드
# 한 번만 다운로드하면 이후에는 자동으로 캐시된 데이터를 사용한다.
nltk.download('stopwords')

# 3️⃣ 예시 문장 설정
text = "This is not only a simple example, but also a very useful one."

# 4️⃣ 모든 단어를 소문자로 변환하고 공백 기준으로 나눈다.
# 대소문자를 통일하면 같은 의미의 단어를 하나로 처리할 수 있다.
tokens = [w.lower() for w in text.split()]

# 5️⃣ NLTK 내장 영어 불용어 목록 불러오기
# stopwords.words('english')는 약 180개 일반 불용어를 포함한다.
base_sw = set(stopwords.words('english'))

# 6️⃣ 화이트리스트(whitelist) 정의
# 부정어(not, no, never)는 의미가 매우 중요하므로 삭제 대상에서 제외한다.
whitelist = {"not", "no", "never"}

# 7️⃣ 실제로 제거할 불용어 집합 정의
# 기본 불용어 목록에서 화이트리스트 단어를 뺀다.
stop_set = base_sw - whitelist

# 8️⃣ 토큰 중에서
#   - 알파벳으로만 이루어진 단어(`isalpha()` 조건)
#   - 불용어 목록에 없는 단어
# 만 남긴다.

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
# 9️⃣ 결과 출력
print(filtered)
# 출력 결과: ['not', 'simple', 'example', 'also', 'useful', 'one']

# 설명:
# - "is", "a", "but" 등 자주 등장하지만 의미가 약한 단어는 제거된다.
# - "not"은 부정 의미를 지니므로 제거되지 않는다.
# - 최종적으로 문장에서 중요한 의미 단어만 남게 되어,
#   이후 분석(예: TF-IDF, 감성 분석 등)의 품질이 높아진다.
