import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ 엑셀 파일 로드
df = pd.read_excel("AirQualityUCI.xlsx", sheet_name=0, engine="openpyxl")

# 2️⃣ 시간 정보 정규화
df["Time"] = (
    df["Time"].astype(str)
    .str.replace(".", ":", n=2, regex=False)
    .str.strip()
    .where(lambda s: s.str.len() > 0, "00:00:00")
)
date_ser = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
date_str = date_ser.dt.strftime("%Y-%m-%d")
df["Datetime"] = pd.to_datetime(date_str + " " + df["Time"], errors="coerce")

# 3️⃣ Datetime 인덱스 설정 및 시간 주기 지정
df = df.set_index("Datetime").sort_index().asfreq("h")

# 4️⃣ 센서 오류값(-200)을 NaN으로 변환
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].replace(-200, np.nan)
df[num_cols] = df[num_cols].infer_objects(copy=False)

# 5️⃣ 결측치 보간 (선형보간 + ffill + bfill)
df_interp = df[num_cols].interpolate(
    method="linear",
    limit_direction="both" # "forward" + "backward" 둘 다 고려해서 양쪽 값의 평균 경로로 보간하는 방식
).ffill().bfill()

# 6️⃣ 시각화 대상 변수 선택
target_col = "CO(GT)"
before = df[target_col]
after = df_interp[target_col]

# 7️⃣ 결측이 있었던 구간 찾기 (시각적 강조용)
mask_nan = before.isna()
if mask_nan.sum() > 0:
    first_nan = mask_nan[mask_nan].index.min()
    last_nan = mask_nan[mask_nan].index.max()
    start_zoom = first_nan - pd.Timedelta(days=2)
    end_zoom = last_nan + pd.Timedelta(days=2)
else:
    start_zoom = df.index.min()
    end_zoom = start_zoom + pd.Timedelta(days=7)

# 8️⃣ 상하 2단 subplot 시각화
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# --- 상단: 보정 전 ---
axes[0].plot(before.index, before, color="gray", alpha=0.6, label="Before Correction")
axes[0].set_title(f"{target_col} - Before Correction")
axes[0].set_ylabel("CO (mg/m³)")
axes[0].grid(alpha=0.3)
axes[0].legend()

# --- 하단: 보정 후 ---
axes[1].plot(after.index, after, color="blue", linewidth=1.5, label="After Interpolation + Fill")
axes[1].set_title(f"{target_col} - After Interpolation + Fill")
axes[1].set_xlabel("Datetime")
axes[1].set_ylabel("CO (mg/m³)")
axes[1].grid(alpha=0.3)
axes[1].legend()

# 확대 영역 설정
axes[0].set_xlim(start_zoom, end_zoom)
axes[1].set_xlim(start_zoom, end_zoom)

plt.tight_layout()
plt.show()
