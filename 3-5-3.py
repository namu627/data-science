import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Theme once
sns.set_theme(style="whitegrid")

# Reproducibility
np.random.seed(42)
n = 150

# 공통적으로 사용할 데이터
df = pd.DataFrame({
    "category": np.random.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25]),
    "value": np.concatenate([
        np.random.normal(0.0, 1.0, 60),
        np.random.normal(0.5, 1.2, 50),
        np.random.normal(-0.3, 0.8, 40)
    ]),
    "group": np.random.choice(["G1", "G2"], size=n),
    "feature1": np.random.normal(0, 1, n),
    "feature2": np.random.normal(1, 1.2, n),
    "feature3": np.random.normal(-0.5, 0.7, n),
})

# 여기에 Seaborn 플롯 별로 사용할 코드를 여기에 붙이기
# Copy & paste codes here!
fig, ax = plt.subplots(figsize=(6, 4))

# Swarmplot 그리기
# Swarmplot: Stripplot과 유사하지만, 점이 겹치지 않게 자동 배치
# sample(120) → 데이터 일부 샘플링 (성능 고려)
sns.swarmplot(
    data=df.sample(120, random_state=0),
    x="category", y="value",
    size=4, linewidth=0,   # 점 크기와 테두리 제거
    ax=ax
)

ax.set_title("Seaborn Swarmplot: Non-overlapping Points per Category")
fig.tight_layout()
plt.show()

# Figure와 Axes 객체 생성
fig, ax = plt.subplots(figsize=(6, 4))

# Barplot 그리기
sns.barplot(
    data=df,           # 사용할 데이터프레임 (df)
    x="category",      # x축에 표시할 변수 (범주형 변수)
    y="value",         # y축에 표시할 변수 (수치형 변수)
    hue="group",       # 범주 그룹을 나눌 변수 (막대를 그룹별로 색상 분리)
    estimator=np.mean, # y값에 대해 어떤 통계량을 계산할지 지정 (기본값은 평균 np.mean)
    ci=95,             # 신뢰구간(confidence interval) 설정 (여기서는 95% CI)
    ax=ax              # 그릴 matplotlib Axes 객체 (subplot에 그릴 때 사용)
)
ax.set_title("Seaborn Barplot: Mean with 95% CI by Category/Group")
plt.show()

# Figure와 Axes 객체 생성
fig, ax = plt.subplots(figsize=(6, 4))

# Countplot 그리기
sns.countplot(
    data=df,        # 사용할 데이터프레임 (df)
    x="category",   # x축에 표시할 변수 (범주형 변수, 카테고리별 빈도를 계산)
    hue="group",    # 그룹 변수 (각 카테고리별로 그룹을 나눠 색상 구분)
    ax=ax           # 그릴 matplotlib Axes 객체 (subplot에 그릴 때 사용)
)

ax.set_title("Seaborn Countplot: Category Frequency")
ax.set_xlabel("Category")
ax.set_ylabel("Count")
plt.show()

# Figure와 Axes 객체 생성
fig, ax = plt.subplots(figsize=(6, 4))

# 상관계수 행렬 계산: 각 변수(feature1, feature2, feature3, value) 간의 상관관계
#   상관계수: -1.0 ~ 1.0 사이 실수값
corr = df[["feature1", "feature2", "feature3", "value"]].corr()

# Heatmap 그리기
sns.heatmap(
    corr,        # 입력 데이터 (상관계수 행렬)
    annot=True,  # 각 셀에 숫자(상관계수 값)를 표시
    fmt=".2f",   # 소수점 둘째 자리까지 표시
    square=True, # 각 셀을 정사각형으로 표시
    ax=ax        # matplotlib Axes 객체 (subplot에 그릴 때 사용)
)

ax.set_title("Seaborn Heatmap: Correlation Matrix")
plt.show()

# 분석할 변수만 선택
sub = df[["feature1", "feature2", "feature3"]]

# Pairplot 그리기
#   - 변수 쌍별로 산점도(scatter plot)를 그림
#   - 대각선(diagonal)에는 히스토그램(histogram)을 표시
#   - corner=True → 하삼각형 부분만 표시 (중복 제거)
#   - corner=False → 상삼각형과 하삼각형 모두 표시 (모든 변수 쌍)
g = sns.pairplot(
    sub,                # 시각화할 DataFrame (수치형 변수만 선택)
    diag_kind="hist",   # 대각선에 표시할 그래프 종류 ("hist" 또는 "kde")
    corner=False        # 변수 쌍을 모두 표시할지 여부 (False면 전체, True면 절반만)
)

# 제목 추가
g.figure.suptitle("Seaborn Pairplot: Feature Relationships", fontsize=14)

# 여백 조정 (top을 줄여 제목 공간 확보)
g.figure.subplots_adjust(top=0.92)
plt.show()

