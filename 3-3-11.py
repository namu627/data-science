import numpy as np
import matplotlib.pyplot as plt

# 난수 데이터 생성
np.random.seed(19680801)
data1 = np.random.randn(100)
data2 = np.random.randn(100)
data3 = np.random.randn(100)

fig, ax = plt.subplots(figsize=(5, 2.7))

# 서로 다른 데이터와 레이블 지정
ax.plot(np.arange(len(data1)), data1, label='data1')   # 기본 선 그래프
ax.plot(np.arange(len(data2)), data2, label='data2')   # 기본 선 그래프
ax.plot(np.arange(len(data3)), data3, 'd', label='data3')  # 다이아몬드 마커 사용

# 범례 표시
ax.legend()

plt.show()
