import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# -----------------------------
# 1. 데이터 생성
# -----------------------------
# X, Y: -3 ~ 3 구간을 128등분한 격자 좌표 생성
X, Y = np.meshgrid(np.linspace(-3, 3, 128), np.linspace(-3, 3, 128))

# Z: X, Y를 이용해 계산된 함수 값 (곡면 형태)
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

# -----------------------------
# 2. Figure와 2x2 Axes 생성
# -----------------------------
fig, axs = plt.subplots(2, 2, layout='constrained', figsize=(8, 8))

# -----------------------------
# 3. pcolormesh()
# -----------------------------
# 격자 데이터(Z)를 색상으로 채워 넣는 함수
# cmap='RdBu_r': 빨강-파랑 반전 팔레트
pc = axs[0, 0].pcolormesh(X, Y, Z, vmin=-1, vmax=1, cmap='RdBu_r')
fig.colorbar(pc, ax=axs[0, 0])  # 색상바(colorbar) 추가
axs[0, 0].set_title('pcolormesh()')

# -----------------------------
# 4. contourf()
# -----------------------------
# 등고선 기반 색상 맵
# levels=np.linspace(-1.25, 1.25, 11): -1.25~1.25 구간을 11단계로 나눔
co = axs[0, 1].contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11))
fig.colorbar(co, ax=axs[0, 1])  # 색상바 추가
axs[0, 1].set_title('contourf()')

# -----------------------------
# 5. imshow() + LogNorm
# -----------------------------
# 2차원 배열을 이미지처럼 표시
# LogNorm: 로그 스케일 색상 맵 → 작은 값과 큰 값을 동시에 표현 가능
pc2 = axs[1, 0].imshow(Z**2 * 100, cmap='plasma',
                       norm=LogNorm(vmin=0.01, vmax=100))
fig.colorbar(pc2, ax=axs[1, 0], extend='both')
axs[1, 0].set_title('imshow() with LogNorm')

# -----------------------------
# 6. scatter()
# -----------------------------
# 점 단위로 데이터 표시, 각 점은 세 번째 데이터(data3)에 따라 색상이 달라짐
data1, data2, data3 = np.random.randn(3, 100)  # x, y, 색상 값
sc = axs[1, 1].scatter(data1, data2, c=data3, cmap='RdBu_r')
fig.colorbar(sc, ax=axs[1, 1])
axs[1, 1].set_title('scatter()')

# -----------------------------
# 7. 최종 출력
# -----------------------------
plt.show()
