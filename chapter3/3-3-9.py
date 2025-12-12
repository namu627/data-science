import numpy as np
import matplotlib.pyplot as plt

# 난수 데이터 준비
np.random.seed(19680801)
data1 = np.random.randn(100)
data2 = np.random.randn(100)
data3 = np.random.randn(100)
data4 = np.random.randn(100)

fig, ax = plt.subplots(figsize=(5, 2.7))

# 각 데이터셋을 서로 다른 마커 스타일로 플로팅
ax.plot(data1, 'o', label='data1')  # 원형 마커
ax.plot(data2, 'd', label='data2')  # 다이아몬드 마커
ax.plot(data3, 'v', label='data3')  # 역삼각형 마커
ax.plot(data4, 's', label='data4')  # 사각형 마커

# 범례 표시
ax.legend()
plt.show()
