from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

# 어간 추출기 (Stemming)
stemmer = PorterStemmer()
# 표제어 추출기 (Lemmatization)
lemmatizer = WordNetLemmatizer()

words = ["studies", "studying", "studied", "better", "running"]

print("단어\t어간추출\t표제어추출")
for w in words:
    stem = stemmer.stem(w)
    lemma = lemmatizer.lemmatize(w, pos=wordnet.VERB)  # 동사 기준으로 변환
    print(f"{w}\t{stem}\t{lemma}")
