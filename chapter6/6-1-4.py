import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])  # x와 완전 선형 관계 (배수)

pearson_r = np.corrcoef(x, y)[0, 1]
cos_sim = cosine_similarity([x], [y])[0, 0]

print("피어슨 상관계수:", pearson_r)
print("코사인 유사도:", cos_sim)
