import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 랜덤 시드 고정
np.random.seed(42)

n_samples = 200

# 기본 변수 3개를 먼저 생성
X1 = np.random.normal(50, 10, n_samples)
X2 = X1 * 0.6 + np.random.normal(0, 5, n_samples)    # X1과 상관 있는 변수
X3 = np.random.normal(30, 5, n_samples)
X4 = X2 * -0.5 + np.random.normal(0, 5, n_samples)   # X2와 음의 상관
X5 = X3 + np.random.normal(0, 2, n_samples)
X6 = np.random.normal(100, 20, n_samples)
X7 = X1 * 0.3 + X3 * 0.4 + np.random.normal(0, 10, n_samples)
X8 = np.random.uniform(0, 1, n_samples)
X9 = X5 * 1.2 + np.random.normal(0, 3, n_samples)
X10 = np.random.normal(0, 1, n_samples)

# 데이터프레임 생성
df_multi = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X4': X4,
    'X5': X5,
    'X6': X6,
    'X7': X7,
    'X8': X8,
    'X9': X9,
    'X10': X10
})

# 상관행렬 계산
corr_matrix = df_multi.corr()

# 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Heatmap of Correlation Matrix (10 Variables)")
plt.tight_layout()
plt.show()
