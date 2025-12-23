from collections import Counter
import matplotlib.pyplot as plt

# 예시 단어 리스트
words = ['data', 'science', 'data', 'machine', 'learning', 'data']

# 각 단어의 등장 횟수를 세기
counter = Counter(words)

# 상위 3개 단어 출력
print(counter.most_common(3))

# Counter 결과를 이용한 시각화
labels, values = zip(*counter.most_common(5))

plt.bar(labels, values)
plt.title("Top 5 Word Frequencies")
plt.xlabel("Words")
plt.ylabel("Count")
plt.show()
