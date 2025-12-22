# ============================================
# Air Quality UCI 시계열 종합 실습 코드
# - 파일: AirQualityUCI.xlsx
# - 전처리: Date/Time → Datetime, asfreq('h'), -200→NaN, 보간+ffill/bfill
# - 분석: 장기추세, 계절별 변화, 일중패턴, 센서 상관(히트맵)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1) 데이터 로드 + Datetime 정규화
# -------------------------------
# (필요 패키지) pip install openpyxl seaborn
df = pd.read_excel("AirQualityUCI.xlsx", sheet_name=0, engine="openpyxl")

# Time이 HH.MM.SS 형태인 경우가 있어 '.' → ':'로 교체, 공백/NaN은 00:00:00으로 대체
df["Time"] = (
    df["Time"].astype(str)
    .str.replace(".", ":", n=2, regex=False)
    .str.strip()
    .where(lambda s: s.str.len() > 0, "00:00:00")
)

# Date를 datetime으로 파싱 후 문자열화(YYYY-MM-DD) → Time과 안전하게 결합
date_ser = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
date_str = date_ser.dt.strftime("%Y-%m-%d")
df["Datetime"] = pd.to_datetime(date_str + " " + df["Time"], errors="coerce")

# Datetime 인덱스 설정 + 시간 주기 명시 (h)
df = df.set_index("Datetime").sort_index().asfreq("h")

# -------------------------------
# 2) 센서 오류값 -200 → NaN, 숫자형 컬럼만 보간
# -------------------------------
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].replace(-200, np.nan)

# 선형보간(양방향) → 남은 NaN은 ffill/bfill로 보완
df[num_cols] = df[num_cols].infer_objects(copy=False)
df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
df[num_cols] = df[num_cols].ffill().bfill()

# 분석에 자주 쓰는 주요 컬럼 후보 (실제 파일에 맞춰 자동 선택)
candidates = ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"]
used = [c for c in candidates if c in df.columns]
if not used:
    raise KeyError("분석용 주요 컬럼(CO(GT), NOx(GT), NO2(GT), C6H6(GT))을 찾을 수 없습니다. 파일 컬럼명을 확인하세요.")

print("[사용 컬럼]", used)

# --------------------------------
# 3) 장기 추세: 30일 이동평균(rolling)
# --------------------------------
# 시간 단위가 시간(h)이므로 30일 ≈ 30*24=720시간 윈도우
roll = df[used].rolling(window=24*30, min_periods=1).mean()

plt.figure(figsize=(12, 5))
for col in used:
    plt.plot(roll.index, roll[col], label=f"{col} (30D MA)")
plt.title("Long-term Trend (30-day Moving Average)")
plt.xlabel("Datetime")
plt.ylabel("Concentration (unit varies)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------
# 4) 계절별 변화: 월별 평균 (Monthly mean)
# --------------------------------
monthly_mean = df[used].resample("M").mean()

plt.figure(figsize=(12, 5))
for col in used:
    plt.plot(monthly_mean.index, monthly_mean[col], marker="o", label=col)
plt.title("Seasonality: Monthly Mean")
plt.xlabel("Month")
plt.ylabel("Concentration (unit varies)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------
# 5) 일중 패턴: 시간대별 평균 (Hour-of-day)
# --------------------------------
# 하루 0~23시 기준 평균
hourly_pattern = df[used].groupby(df.index.hour).mean()

plt.figure(figsize=(12, 5))
for col in used:
    plt.plot(hourly_pattern.index, hourly_pattern[col], marker="o", label=col)
plt.xticks(range(0, 24))
plt.title("Daily Pattern: Mean by Hour of Day")
plt.xlabel("Hour of Day (0–23)")
plt.ylabel("Concentration (unit varies)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------
# 6) 센서 간 상관관계: 히트맵
# --------------------------------
# 수치형 전체 상관 또는 주요컬럼만
corr = df[used].corr(method="pearson")

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, cbar=True)
plt.title("Sensor/Target Correlation (Pearson)")
plt.tight_layout()
plt.show()

# --------------------------------
# 7) 보너스: 결과 요약 출력
# --------------------------------
print("\n[월별 평균 상위 5행]")
print(monthly_mean.head())
print("\n[시간대별 평균 상위 5행]")
print(hourly_pattern.head())
print("\n[상관행렬]")
print(corr)
