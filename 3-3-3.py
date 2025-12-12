'''Figure–Axes–Axis 구조 다이어그램'''

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

# Figure 영역
fig_box = Rectangle((0.5, 0.5), 9, 6, fill=False, linewidth=2)
ax.add_patch(fig_box)
ax.text(0.6, 6.7, "Figure", fontsize=12, va="bottom")

# Axes 영역
axes_box = Rectangle((1.3, 1.3), 7.4, 4.6, fill=False, linewidth=2)
ax.add_patch(axes_box)
ax.text(1.4, 5.9, "Axes", fontsize=12, va="bottom")

# 축 레이블 표시
ax.text(8.5, 1.1, "x-axis (Axis)", fontsize=10)
ax.text(1.0, 5.0, "y-axis (Axis)", fontsize=10, rotation=90)

# 눈금 흉내
for t in np.linspace(1.5, 8.3, 6):
    ax.plot([t, t], [1.3, 1.4])
for t in np.linspace(1.7, 5.7, 5):
    ax.plot([1.3, 1.4], [t, t])

# Axes 내부 데이터 예시
xs = np.linspace(1.5, 8.5, 100)
ys = 3 + 1.2*np.sin((xs-1.5)/7.0 * 2*np.pi)
ax.plot(xs, ys)

plt.show()
