import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Seaborn 내장 데이터 불러오기
flights = sns.load_dataset("flights")   # 항공 승객 수 데이터

# flights 데이터를 wide-format으로 변환
flights_wide = flights.pivot(
    index="year",       # 행 인덱스: 연도
    columns="month",    # 열: 월
    values="passengers" # 값: 승객 수
)

# 변환 결과 예시:
#   DataFrame은 각 열이 한 달을 나타내며, 행은 연도별 승객 수를 기록
#   "월별 데이터의 연도별 흐름"을 동시에 파악

# month   January  February  March  ...  November  December
# year
# 1949        112       118    132  ...       104       118
# 1950        115       126    141  ...       114       140
# ...

# subplot 2행 1열 배치
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# (1) Long-format 데이터 시각화
# 시계열 데이터 시각화
# relplot: "relationship plot"의 줄임말
#           산점도나 선 그래프를 그릴 때 사용되는 Seaborn의 범용 함수
sns.lineplot(
    data=flights,      # 사용할 데이터프레임 (flights: 1949~1960 항공 여객 데이터)
    x="month",         # X축: 월(month) → 1월 ~ 12월
    y="passengers",    # Y축: 승객 수(passengers)
    hue="year",        # 색상 구분: 연도별(year)로 색을 다르게 표시
    marker="o",        # 데이터 지점마다 동그란 마커(o)를 추가하여 값이 더 잘 보이게 함
    ax=axes[0],        # 첫 번째 subplot(axes[0])에 그래프 그리기
)
axes[0].set_title("Monthly Air Passengers by Year (Long-format)")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Passengers")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(True)

# (2) Wide-format 데이터 시각화
sns.lineplot(
    data=flights_wide,
    dashes=False,  # 각 월별 시계열 구분이 명확하게 보이도록 설정
    ax=axes[1]
)
axes[1].set_title("Air Passengers Trend by Month (Wide-format)")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Passengers")
axes[1].grid(True)

# 레이아웃 조정
plt.tight_layout()

# 화면에 출력
plt.show()
