import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 예시 데이터 (월별 대기오염 농도)
date_rng = pd.date_range(start='2020-01', end='2023-01', freq='ME')
data = {
    'PM10': [80,75,70,65,90,100,95,85,80,75,70,65,
             90,100,95,85,80,75,70,65,90,100,95,85,
             80,75,70,65,90,100,95,85,80,75,70,65]
}
df = pd.DataFrame(data, index=date_rng)

# -------------------------------
# 1️⃣ 가법모형 (Additive Model)
# -------------------------------
result_add = seasonal_decompose(df['PM10'], model='additive')
fig1 = result_add.plot()
plt.show()

# -------------------------------
# 2️⃣ 승법모형 (Multiplicative Model)
# -------------------------------
result_mul = seasonal_decompose(df['PM10'], model='multiplicative')
fig2 = result_mul.plot()
plt.show()
