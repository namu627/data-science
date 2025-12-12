import pandas as pd
import matplotlib.pyplot as plt

# 샘플 시계열 데이터 생성
dates = pd.date_range("2025-03-01", periods=30)

# 일반적으로 데이터를 수집하는 과정을 거쳐야 하지만, 우리는 있다고 가정함.
visitors = [100,120,115,130,150,140,160,155,170,180,
            175,190,200,210,220,230,225,240,250,260,
            270,265,280,290,300,310,305,320,330,340]
df = pd.DataFrame(
    {
        "date": dates,
        "visitors": visitors
    }
).set_index("date")

# 이동평균 계산
# "visitors" 열에서 7일 이동평균(rolling mean)을 계산하여 새로운 열 "7d_avg"에 저장
#   - df["visitors"] : 웹사이트의 일별 방문자 수 데이터
#   - .rolling(7)    : 현재 행을 기준으로 이전 6일 + 현재일까지 총 7일 구간의 데이터를 묶음
#   - .mean()        : 묶인 7일 구간의 평균을 계산
# 각 날짜별로 그 날을 포함한 7일간의 평균 방문자 수가 "7d_avg" 열에 기록됨
df["7d_avg"] = df["visitors"].rolling(7).mean()

# 시각화
plt.plot(df.index, df["visitors"], label="Daily Visitors")
plt.plot(df.index, df["7d_avg"], label="7-day Moving Avg", linewidth=2)
plt.title("Website Visitors Trend")
plt.xlabel("Date")
plt.ylabel("Visitors")
plt.legend()
plt.show()
