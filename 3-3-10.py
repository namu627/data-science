import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성: 평균 115, 표준편차 15를 따르는 정규분포 난수 10000개
mu, sigma = 115, 15
x = mu + sigma * np.random.randn(10000)

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

# 히스토그램 그리기
n, bins, patches = ax.hist(x, 50, density=True, facecolor='C0', alpha=0.75)

# 축 레이블, 제목, 텍스트 추가
ax.set_xlabel('Length [cm]')
ax.set_ylabel('Probability')
ax.set_title('Aardvark lengths\n(not really)')
ax.text(75, 0.025, r'$\mu=115,\ \sigma=15$')  # 좌표 (75, 0.025)에 텍스트 추가

# 축 범위와 격자 설정
ax.axis([55, 175, 0, 0.03])
ax.grid(True)

plt.show()
