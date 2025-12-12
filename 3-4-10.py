import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 가상 데이터 생성 
# 난수 시드 고정 → 항상 동일한 값 생성
np.random.seed(0)

days = ["Sun", "Sat", "Thur", "Fri"]
times = ["Lunch", "Dinner"]

data = []
for day in days:
  for time in times:
    # 각 조합에 대해 총 30개 샘플 생성
    bills = np.random.normal(
      loc=20 if time=="Lunch" else 25,  # 정규분포의 평균
      scale=8,                          # 정규분포의 표준편차
      size=30                           # 샘플 개수
    )
    for b in bills:
      data.append({"day": day, "time": time, "total_bill": b})

df = pd.DataFrame(data)

# Boxplot 그리기
sns.boxplot(
  data=df, 
  x="day",          # x축: 요일
  y="total_bill",   # y축: 총 금액
  hue="time",       # 'time' 기준으로 색상 구분: 점심/저녁
  order=days        # x축 순서 지정
)

plt.title("Total Bill by Day and Time (Fake Data)")
plt.show()
