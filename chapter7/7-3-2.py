import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) 엑셀 파일 로드 (openpyxl 패키지가 필요함)
# - AirQualityUCI.xlsx 파일을 불러옴
# - sheet_name=0 : 첫 번째 시트를 읽음
# - engine="openpyxl" : .xlsx 파일을 처리하는 데 필요한 엔진 지정
df = pd.read_excel(
    "AirQualityUCI.xlsx",
    sheet_name=0,
    engine="openpyxl"
)

# 2) Date/Time 정규화
#   - Date: 어떤 타입이든 datetime으로 변환하여 날짜 형식을 통일
#   - dayfirst=True : 날짜가 'DD/MM/YYYY' 형식일 때 올바르게 인식되도록 지정
#   - errors="coerce" : 변환 불가능한 값은 NaT(Not a Time)으로 처리하여 오류 방지
date_ser = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# datetime 타입의 날짜를 문자열 'YYYY-MM-DD' 형식으로 변환
# 문자열로 만들어 두면 나중에 Time과 안전하게 결합할 수 있음
date_str = date_ser.dt.strftime("%Y-%m-%d")

# Time 컬럼 정규화
#   - Time 값이 HH.MM.SS 형식으로 되어 있으므로, 점(.)을 콜론(:)으로 바꿈
#   - astype(str) : 숫자나 결측값 등 모든 값을 문자열로 통일
#   - str.replace(".", ":", n=2, regex=False) : 앞에서 두 번만 '.'을 ':'로 교체 (HH.MM.SS → HH:MM:SS)
#   - str.strip() : 앞뒤 공백을 제거
time_str = (
    df["Time"]
    .astype(str)
    .str.replace(".", ":", n=2, regex=False)
    .str.strip()
)

# Time 문자열 중 길이가 0인 항목(즉, 비어 있는 값)을 찾아서
# "00:00:00"으로 채워주는 코드임.
#   -> Time 정보가 비어 있으면 자정(00시)으로 대체하여 Datetime 생성 시 오류를 방지
time_str = time_str.where(time_str.str.len() > 0, "00:00:00")

# 3) 최종 Datetime 생성 (문자열 결합 후 파싱)
df["Datetime"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")

# 4) 인덱스/주기 설정
df = df.set_index("Datetime").sort_index().asfreq("H")  # 시간 단위(Hourly)

# 5) 센서 오류값 -200을 결측치로 처리 (UCI Air Quality 관례)
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].replace(-200, np.nan)

# 6) 확인
print(df.tail())
print("\n결측치 개수:")
print(df.isnull().sum())

# 7) 간단 시각화: CO(GT) 농도 시계열 그래프
if "CO(GT)" in df.columns:
    ax = df["CO(GT)"].plot(figsize=(12, 4), title="CO(GT) Concentration Over Time")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("CO (mg/m^3)")
    plt.tight_layout()
    plt.show()
else:
    print("Column 'CO(GT)' not found in the DataFrame.")
