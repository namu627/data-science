import pandas as pd
import matplotlib.pyplot as plt

# 1) 데이터 로드
df = pd.read_csv("train.csv")  # Kaggle에서 내려받은 train.csv 사용

# 2) 빠른 점검
print("shape:", df.shape)              # 데이터 크기 확인
print("null counts:\n", df.isnull().sum())  # 결측치 개수 확인

# 3) 전처리
df["Age"] = df["Age"].fillna(df["Age"].median())     # Age는 중앙값으로 대치
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # Embarked는 최빈값으로 대치

# 4) GroupBy 예시: 성별별 평균 생존률
survival_by_sex = df.groupby("Sex")["Survived"].mean()
print("\nGroupBy - Survival rate by Sex:\n", survival_by_sex)

# 5) Pivot Table 예시: 성별×좌석등급 평균 생존률
pivot_survival = pd.pivot_table(
    df, values="Survived", index="Sex", columns="Pclass", aggfunc="mean"
)
print("\nPivot Table - Survival rate by Sex and Pclass:\n", pivot_survival)

# 6) 시각화 (영문 라벨 권장)
plt.figure()
df["Age"].plot(kind="hist", bins=20, title="Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("./imgs/chap_02/age_distribution.png", dpi=200)

plt.figure()
survival_by_sex.plot(kind="bar", title="Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("./imgs/chap_02/survival_by_sex.png", dpi=200)

# 7) 인사이트 출력
insights = [
    "Female passengers show significantly higher survival rates",
    "First class survival rate is much higher than third class",
    "Mean fare differs by Embarked port, suggesting socio-economic patterns"
]
print("\nInsights:")
for i in insights:
    print("-", i)
