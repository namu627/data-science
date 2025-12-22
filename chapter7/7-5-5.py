# File: 08_stationarity_differencing_practice.py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 1. 데이터 로드 (AirPassengers 데이터셋)
# 이 데이터는 1949~1960년 국제 항공 여객 수 (월별, 단위: 1000명)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
df.columns = ['Passengers']  # 컬럼명 통일
print(df.head())

# 2. 정상성 검정 (ADF Test)
adf_result = adfuller(df["Passengers"])
print("ADF Statistic:", round(adf_result[0], 3))
print("p-value:", round(adf_result[1], 4))

if adf_result[1] < 0.05:
    print("✅ 정상 시계열 (차분 불필요)")
else:
    print("⚠️ 비정상 시계열 (차분 필요)")

# 3. 차분 수행
df["diff1"] = df["Passengers"].diff()
df["diff_seasonal"] = df["Passengers"].diff(12)

# 4. 시각화 비교
fig, axes = plt.subplots(3, 1, figsize=(10,8), sharex=True)

axes[0].plot(df["Passengers"], label="Original", color='blue')
axes[0].set_title("Original Time Series")
axes[0].legend()

axes[1].plot(df["diff1"], label="1st Difference", color='orange')
axes[1].set_title("1st Difference (Trend Removed)")
axes[1].legend()

axes[2].plot(df["diff_seasonal"], label="Seasonal Difference", color='green')
axes[2].set_title("Seasonal Difference (Period=12)")
axes[2].legend()

plt.tight_layout()
plt.show()
