import numpy as np
import matplotlib.pyplot as plt

# 난수 데이터 생성
np.random.seed(19680801)
data1 = np.random.randn(100)
data2 = np.random.randn(100)

# 색상 매핑 기준 (0 ~ 1 범위)
colors = np.linspace(0, 1, 100)

# 산점도 그리기
fig, ax = plt.subplots(figsize=(5, 2.7))
scatter = ax.scatter(
    data1, data2,
    c=colors,          # 색상 값 지정
    cmap='plasma',     # plasma 팔레트 적용
    s=50,              # 마커 크기
    edgecolor='k'      # 테두리 색상 (검정)
)

# 컬러바 추가
plt.colorbar(scatter, ax=ax, label="Color scale (plasma)")

plt.show()
