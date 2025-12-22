import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 준비
df = pd.read_excel("AirQualityUCI.xlsx", sheet_name=0, engine="openpyxl")
df["Time"] = df["Time"].astype(str).str.replace(".", ":", n=2, regex=False).str.strip()
date_ser = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
date_str = date_ser.dt.strftime("%Y-%m-%d")
df["Datetime"] = pd.to_datetime(date_str + " " + df["Time"], errors="coerce")
df = df.set_index("Datetime").sort_index().asfreq("h")

# 2. 숫자형 컬럼만 추출 및 센서 오류값(-200) 제거
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].replace(-200, np.nan)

# 3. 대상 컬럼 선택
target_col = "CO(GT)"
series = df[target_col].dropna()

# 4. IQR 계산
Q1 = series.quantile(0.25)
Q3 = series.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 5. 이상치 탐지
outliers = series[(series < lower_bound) | (series > upper_bound)]

print(f"이상치 개수: {len(outliers)}")
print(f"상한: {upper_bound:.2f}, 하한: {lower_bound:.2f}")

# 6. 시각화
plt.figure(figsize=(12,5))
plt.plot(series.index, series, label="Original", color="gray")
plt.scatter(outliers.index, outliers, color="red", label="Outliers")
plt.title("Outlier Detection using IQR Method")
plt.xlabel("Datetime")
plt.ylabel("CO (mg/m³)")
plt.legend()
plt.tight_layout()
plt.show()

median = series.median()
series_corrected = series.copy()
series_corrected[outliers.index] = median

plt.figure(figsize=(12,5))
plt.plot(series.index, series, label="Before Correction", color="gray", alpha=0.6)
plt.plot(series_corrected.index, series_corrected, label="After Correction", color="blue", linewidth=1.2)
plt.scatter(outliers.index, outliers, color="red", label="Detected Outliers")
plt.title("Outlier Correction (Median Replacement)")
plt.xlabel("Datetime")
plt.ylabel("CO (mg/m³)")
plt.legend()
plt.tight_layout()
plt.show()