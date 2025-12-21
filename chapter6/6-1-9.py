import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 기본 확인
DATA_PATH = "./boston.csv"  # 파일명은 데이터셋에 따라 조정

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"'{DATA_PATH}' 파일을 찾을 수 없습니다. 데이터를 다운로드하고 경로를 확인하세요.")

df = pd.read_csv(DATA_PATH)

print("데이터 크기:", df.shape)
print("\n컬럼 타입 요약:")
print(df.dtypes.value_counts())

print("\n결측치 상위 컬럼:")
print(df.isnull().sum().sort_values(ascending=False).head())

print("\n수치형 변수 기술통계 (상위 5):")
print(df.describe().T.head())

# 2. 수치형 변수만 상관행렬 계산
target_col = "MEDV"
if target_col not in df.columns:
    raise KeyError(f"'{target_col}' 컬럼이 없습니다. 데이터셋을 확인하세요.")

num_df = df.select_dtypes(include=[np.number])
corr_all = num_df.corr(method="pearson")

target_corr = corr_all[target_col].drop(labels=[target_col]).sort_values(ascending=False)
top_feats = target_corr.head(10).index.tolist()
corr_top = corr_all.loc[top_feats + [target_col], top_feats + [target_col]]

print("\n{}와 상관 상위 10개 변수:".format(target_col))
print(target_corr.head(10))

# 3‑1. 전체 상관행렬 히트맵
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_all,
    cmap="coolwarm",
    center=0,
    square=True,
    cbar=True,
    xticklabels=False,
    yticklabels=False
)
plt.title("Correlation Matrix (All Numeric Features)")
plt.tight_layout()
plt.show()

# 3‑2. 상위 변수 + Target 히트맵
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_top,
    cmap="coolwarm",
    center=0,
    square=True,
    annot=True, fmt=".2f",
    cbar=True
)
plt.title(f"Correlation Matrix (Top variables with {target_col})")
plt.tight_layout()
plt.show()
