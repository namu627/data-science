# ==========================================
# 🇰🇷 대통령 연설문 단어 빈도 분석 및 시각화
# ==========================================

from konlpy.tag import Okt, Mecab
from collections import Counter
import re
import matplotlib.pyplot as plt

# 한글 폰트 설정 (윈도우 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 기본 폰트
plt.rcParams['axes.unicode_minus'] = False

# 1) 형태소 분석기 준비
# - Okt(Open Korean Text)는 한국어 형태소 분석기 중 하나로,
#   SNS/연설문 같은 자연스러운 문체에서 좋은 성능을 보인다.
okt = Okt()

# 2) 연설문 불러오기
# - speech.txt 파일에 대통령 연설문 전문이 저장되어 있다고 가정한다.
# - UTF-8 인코딩을 사용 (윈도우에서 인코딩 오류가 날 경우 'cp949'로 변경 가능)
with open("president_speech.txt", encoding="utf-8") as f:
    text = f.read()

# 3) 정규화: 한글과 공백만 남기기
# - [^가-힣\s] : 한글(가~힣)과 공백(\s)을 제외한 모든 문자를 찾아 공백으로 대체
# - 여러 공백을 하나로 줄이고, 문장 앞뒤 공백을 제거한다.
text = re.sub(r"[^가-힣\s]", " ", text)
text = re.sub(r"\s+", " ", text).strip()

# 4) 형태소 분석 & 명사 추출
# - okt.nouns()는 문장에서 명사만 추출해 리스트 형태로 반환한다.
tokens = okt.nouns(text)

# 5) 1글자 명사 제거
# - 조사·단음절 등 의미가 약한 단어 제거
tokens = [w for w in tokens if len(w) > 1]

# 6) 불용어 정의
# - 문서 분석 목적상 불필요하거나 너무 일반적인 단어를 제외한다.
# - 필요에 따라 도메인별로 불용어를 추가/삭제할 수 있다.
stopwords = {"국민", "정부", "대한민국", "대통령", "우리", "것", "수"}

# 7) 불용어 제거
filtered = [w for w in tokens if w not in stopwords]

# 8) 단어 빈도 계산
# - Counter는 각 단어의 등장 횟수를 세어 딕셔너리 형태로 반환한다.
counter = Counter(filtered)

# 9) 상위 10개 단어 추출
top10 = counter.most_common(10)

# 10) 결과 출력
print("상위 10개 명사:", top10)

# ------------------------------------------
# 11) 시각화: 막대 그래프
# ------------------------------------------

# Counter 결과를 (단어, 빈도) 형태로 분리
labels, values = zip(*top10)

# 그래프 크기 설정
plt.figure(figsize=(8, 5))

# 막대그래프 그리기
plt.bar(labels, values, color="skyblue")

# 그래프 제목과 축 라벨
plt.title("대통령 연설문 Top 10 단어 빈도", fontsize=14)
plt.xlabel("단어", fontsize=12)
plt.ylabel("빈도수", fontsize=12)

# x축 글자 겹침 방지
plt.xticks(rotation=45)

# 그래프 여백 조정
plt.tight_layout()

# 그래프 출력
plt.show()
