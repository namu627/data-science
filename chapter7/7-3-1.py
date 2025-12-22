# ==========================================
# 📊 AirPassengers 시계열 데이터 전처리
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ CSV 파일 불러오기
# Kaggle에서 받은 AirPassengers.csv 파일을 불러온다.
# 데이터에는 'Month'와 'Passengers' 두 개의 열이 포함되어 있다.
df = pd.read_csv("AirPassengers.csv")

# 2️⃣ 데이터 상위 5개 확인
print("원본 데이터 (상위 5행):")
print(df.head())

# 3️⃣ 'Month' 열을 날짜(datetime) 형식으로 변환
# 문자열 형식의 날짜를 datetime 형식으로 바꿔야
# 시계열 연산(예: 차분, 이동평균 등)을 수행 가능
df["Month"] = pd.to_datetime(df["Month"])

# 4️⃣ 'Month'를 인덱스로 설정하고, 월 단위(Month Start) 시계열로 지정
# asfreq("MS")는 "Month Start" 빈도로 시계열 주기 설정
df = df.set_index("Month").asfreq("MS")

# 5️⃣ 변환된 데이터 확인
print("\n시계열 인덱스 적용 후 데이터:")
print(df.head())

# 6️⃣ 데이터 기본 통계 요약
print("\n📈 데이터 통계 요약:")
print(df.describe())

# 7️⃣ 결측치 확인
print("\n결측치 확인:")
print(df.isnull().sum())

# 8️⃣ 시각화: 월별 여객 수 변화 추이
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["#Passengers"], color="royalblue", linewidth=2)
plt.title("Monthly Number of Air Passengers (1949–1960)", fontsize=13)
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 9️⃣ 간단한 통계 해석 출력
print("\n📊 해석:")
print("• 데이터 기간:", df.index.min().strftime("%Y-%m"), " ~ ", df.index.max().strftime("%Y-%m"))
print("• 전체 관측치 수:", len(df))
print("• 월별 평균 여객 수:", round(df['#Passengers'].mean(), 2))
print("• 최대 여객 수:", df['#Passengers'].max())
print("• 최소 여객 수:", df['#Passengers'].min())
