# 목적: PMF(주사위)와 PDF(정규분포) 비교 그림 저장
# 환경: Python 3.10+, matplotlib, scipy 필요

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -------- PMF (주사위 예시) --------
x_pmf = np.arange(1, 7)
pmf = np.ones_like(x_pmf) / 6  # 균등 분포 (1/6)

# -------- PDF (정규분포 예시) --------
x_pdf = np.linspace(-4, 10, 500)
mu, sigma = 3, 1.5  # 평균과 표준편차
pdf = norm.pdf(x_pdf, mu, sigma)

# -------- 시각화 --------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# PMF (이산형 확률변수: 주사위)
axes[0].bar(x_pmf, pmf, color="skyblue", edgecolor="black")
axes[0].set_xticks(x_pmf)
axes[0].set_title("PMF - Dice Roll (Discrete)", fontsize=12)
axes[0].set_xlabel("Dice Outcome")
axes[0].set_ylabel("Probability")

# PDF (연속형 확률변수: 정규분포)
axes[1].plot(x_pdf, pdf, color="tomato", linewidth=2)
axes[1].fill_between(x_pdf, pdf, color="tomato", alpha=0.3)
axes[1].set_title("PDF - Normal Distribution (Continuous)", fontsize=12)
axes[1].set_xlabel("x")
axes[1].set_ylabel("Density")

plt.tight_layout()

# -------- 이미지 저장 --------
plt.savefig("pmf_pdf_example.png", dpi=300)
plt.show()

print("'pmf_pdf_example.png' 파일이 저장되었습니다.")
