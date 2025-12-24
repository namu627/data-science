# ============================================
# 🎆 워드클라우드 실습: 명사 기반 시각화
# ============================================

from konlpy.tag import Okt
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1️⃣ 텍스트 파일 읽기
# UTF-8 인코딩으로 한글 깨짐 방지
with open("independent_day.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2️⃣ 형태소 분석기 초기화 (Okt)
# Okt는 한국어 문장에서 명사, 동사, 형용사 등을 구분해줌
okt = Okt()

# 3️⃣ 명사만 추출
# 예: ["독립", "기념일", "국가", "대한민국", ...]
nouns = okt.nouns(text)

# 4️⃣ 불용어(Stopwords) 제거
# 분석 목적과 상관없는 조사, 접속사 등은 제거
stopwords = ["것", "수", "등", "들", "그", "그리고", "이", "저", "제", "우리", "대한"]
filtered_nouns = [word for word in nouns if word not in stopwords and len(word) > 1]

# 5️⃣ 워드클라우드를 위한 문자열 결합
text_for_wc = " ".join(filtered_nouns)

# 6️⃣ 한글 폰트 경로 설정 (운영체제별 경로 예시)
# Windows: C:/Windows/Fonts/malgun.ttf
# macOS: /System/Library/Fonts/AppleGothic.ttf
# Linux/Colab: /usr/share/fonts/truetype/nanum/NanumGothic.ttf
font_path = "C:/Windows/Fonts/malgun.ttf"

# 7️⃣ 워드클라우드 생성
# ============================================
# WordCloud 파라미터 사용 가이드
# --------------------------------------------
# font_path : 워드클라우드에서 사용할 폰트 파일 경로 (한글 필수)
#   - 미설정 시 한글이 □□□로 깨짐
#   - OS별 예시:
#       Windows → "C:/Windows/Fonts/malgun.ttf"  (맑은 고딕)
#       macOS   → "/System/Library/Fonts/AppleGothic.ttf" (애플고딕)
#       Linux   → "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" (나눔고딕)
#   - Colab에서는 다음으로 설치 가능:
#       !apt-get install fonts-nanum -y && fc-cache -fv
#
# background_color : 배경색 (문자열/HEX)
#   - 예: "white", "black", "#f5f5f5"
#   - 발표/리포트용은 보통 "white" 선호
#
# width, height : 출력 이미지 해상도(픽셀)
#   - 넓을수록 글자 디테일이 좋아짐(파일 저장에 유리)
#   - 예: width=1200, height=800
#
# max_words : 표시할 최대 단어 수
#   - 상위 빈도 단어 n개만 시각화
#   - 너무 크게 잡으면 글자 겹침/가독성 저하 → 100~200 권장
#
# colormap : 색상 팔레트 (matplotlib colormap 이름)
#   - 예: "tab10", "Set2", "viridis", "plasma", "coolwarm"
#   - 참고: https://matplotlib.org/stable/users/explain/colors/colormaps.html
#
# prefer_horizontal : 수평 배치 비율(0~1)
#   - 기본 0.9 (대부분 가로 배치)
#   - 세로 단어가 많아도 되면 0.5 정도로 조정
#
# scale : 렌더링 스케일(배율)
#   - 값↑ → 더 선명한 이미지(대신 속도/메모리 ↑)
#   - 저장용 이미지는 scale=2~3 고려
#
# random_state : 난수 시드(재현성)
#   - 동일 텍스트라 해도 시드가 같으면 배치/색상 패턴이 동일
#   - 학습/보고서는 고정 권장(예: 42)
#
# mask : 마스킹 이미지(ndarray)
#   - 특정 모양(로고/학과 엠블럼)으로 단어 배치
#   - 예: mask = plt.imread("logo_mask.png")
#
# generate(text) / generate_from_frequencies(freq_dict)
#   - text: 공백으로 구분된 단어 문자열
#   - freq_dict: {"단어": 빈도, ...} 형태 (CountVectorizer, TF-IDF 등과 연계 편리)
# ============================================

wc = WordCloud(
    font_path=font_path,          # 한글 폰트 경로
    background_color="white",     # 배경색
    width=800,                    # 이미지 가로 크기(px)
    height=500,                   # 이미지 세로 크기(px)
    max_words=150,                # 최대 단어 수
    colormap="tab10",             # 색상 팔레트
    prefer_horizontal=0.9,        # 수평 배치 비율
    scale=1.0,                    # 렌더링 스케일(선명도)
    random_state=42               # 재현성(결과 고정)
    # mask=mask                   # (선택) 마스킹 이미지 사용 시 주석 해제
).generate(text_for_wc)           # 또는 .generate_from_frequencies(freq_dict)


# 8️⃣ 시각화
plt.figure(figsize=(10, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud (Noun-based) - Independence Day Speech", fontsize=16)
plt.show()
