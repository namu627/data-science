import nltk

# 최초 한 번만 실행하면 되는 부분 입니다.
#   NLTK 데이터 다운로드
#   이후 실행 시 download()는 불필요합니다.

# nltk.download('punkt')       # 토큰화용 데이터
# nltk.download('punkt_tab')   # 탭 구분 토큰화용 데이터
# nltk.download('stopwords')   # 불용어 리스트
# nltk.download('wordnet')     # 표제어 추출용 데이터

text = "Data Science is an exciting field."
tokens = nltk.word_tokenize(text)

print(tokens)
