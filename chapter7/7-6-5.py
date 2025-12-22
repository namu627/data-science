# File: 09_sarima_forecasting_example_v2.py
# SARIMA 모델을 이용한 항공 여객 수 예측 실습 (경고 제거 버전)
# ------------------------------------------------------------
# 이 예제는 AirPassengers 데이터셋을 사용하여
# 월별 항공 여객 수를 예측하는 SARIMA 모델을 실습하는 코드이다.
# ------------------------------------------------------------

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# (1) 데이터 로드 및 전처리
# ------------------------------------------------------------

# AirPassengers 데이터셋: 1949~1960년 국제 항공 여객 수 (월별)
# 데이터 출처: https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

# CSV 파일을 불러오며, 'Month' 열을 날짜(Datetime)로 변환하고
# 이를 인덱스로 설정하여 시계열 형태로 구성
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# 컬럼명을 'Passengers'로 변경 (기존 컬럼명: 'Passengers'와 동일하지만 명시적으로 지정)
df.columns = ['Passengers']

# 인덱스 주기(frequency) 명시
# "MS"는 Month Start (매월 1일) 주기를 의미하며,
# statsmodels에서 주기 경고(ValueWarning)를 제거하기 위해 반드시 설정 필요
df = df.asfreq('MS')        # 또는 df.index.freq = 'MS' 도 가능

# 데이터 확인 (앞부분 5개 행)
print(f"✅ Data frequency set to: {df.index.freqstr}")
print(df.head())

# ------------------------------------------------------------
# (2) SARIMA 모델 정의
# ------------------------------------------------------------
# SARIMA(p, d, q) (P, D, Q, s)
#
# ─ 비계절(Non-seasonal) 차수: order = (p, d, q)
#   • p (AR order): 비계절 자기회귀 차수
#       - 현재 시점 y_t가 과거의 y_{t-1}, y_{t-2}, ... y_{t-p}까지 
#         몇 단계의 "값"에 의존하는지
#       - p가 크면 과거 값들을 많이 참조 → 과적합 위험 ↑, PACF 절단점으로 대략 판단
#   • d (Integration, differencing): 비계절 차분 횟수
#       - 추세(비정상성) 제거를 위해 몇 번 차분할지 (보통 0 또는 1, 드물게 2)
#       - 과도한 차분은 정보 손실/과분산 유의
#   • q (MA order): 비계절 이동평균 차수
#       - 현재 시점 y_t가 과거의 오차(잔차) e_{t-1}, e_{t-2}, ... e_{t-q}에 
#         얼마만큼 의존하는지
#       - q는 ACF 절단점으로 대략 판단
#
# ─ 계절(Seasonal) 차수: seasonal_order = (P, D, Q, s)
#   • P (Seasonal AR order): 계절 자기회귀 차수
#       - y_t가 계절 간격 s만큼 떨어진 
#         과거 값(y_{t-s}, y_{t-2s}, …, y_{t-Ps})에 의존하는 정도
#       - 예) s=12(월별 데이터)일 때, 1년 전/2년 전 같은 달의 값 의존
#   • D (Seasonal differencing): 계절 차분 횟수
#       - 계절적 평균 이동을 제거하기 위해 몇 번 계절 차분할지 (보통 0 또는 1)
#       - D=1이면 ∇_s y_t = y_t − y_{t−s} (예: 이번 달 − 작년 같은 달)
#   • Q (Seasonal MA order): 계절 이동평균 차수
#       - 현재 y_t가 계절 간격 s로 떨어진 과거 오차(e_{t-s}, e_{t-2s}, …)에 
#         의존하는 정도
#   • s (Seasonal period): 계절 주기 길이
#       - 월별 데이터의 연간 주기: s=12
#       - 분기 데이터의 연간 주기: s=4
#       - 일별 데이터의 주간 주기: s=7 등
#
# ─ 기타 옵션
#   • enforce_stationarity=False
#       - 모수 추정 시 “정상성(Stationarity) 제약”을 강제로 두지 않음
#       - 데이터가 약간 비정상이어도 추정이 수렴하도록 유연성 부여
#   • enforce_invertibility=False
#       - “가역성(Invertibility) 제약”을 강제로 두지 않음
#       - MA 성분이 경계에 가까운 경우에도 추정 허용(수렴성/안정성 트레이드오프)
#
# 예시: (1,1,1)(1,1,1,12)는
#   → 비계절 AR(1) + 비계절 차분 1회(d=1) + 비계절 MA(1)
#   → 12개월 주기의 계절 AR(1) + 계절 차분 1회(D=1) + 계절 MA(1)
#   → 즉, 추세와 12개월 계절성이 모두 존재한다고 보고 이를 동시에 모형화

non_seasonal_order = (1, 1, 1)      # (p, d, q)
seasonal_order = (1, 1, 1, 12)      # (P, D, Q, s)  ← 월별 데이터의 연간 주기

model = SARIMAX(
    endog=df["Passengers"],          # 종속변수 (시계열)
    order=non_seasonal_order,        # 비계절 차수 (p, d, q)
    seasonal_order=seasonal_order,   # 계절 차수 (P, D, Q, s)
    enforce_stationarity=False,      # 정상성 강제 X (경계 모수 허용)
    enforce_invertibility=False,     # 가역성 강제 X (경계 모수 허용)
)

# ------------------------------------------------------------
# (3) 모델 학습 (Fitting)
# ------------------------------------------------------------

# 모델 학습을 수행하면 내부적으로 MLE(Maximum Likelihood Estimation)를 통해
# AR, MA, 계절 항의 계수를 추정함
result = model.fit()

# ------------------------------------------------------------
# 결과 요약 출력 (모델 통계 요약표)
# ------------------------------------------------------------
# .summary() 출력 항목 주요 해석:
#
# [1] AIC (Akaike Information Criterion)
#     - 모델의 적합도와 복잡도(파라미터 수)를 동시에 고려하는 지표
#     - 값이 작을수록 데이터에 더 잘 맞고 과적합 위험이 낮음
#     - 여러 모델 비교 시, AIC가 가장 작은 모델 선택
#
# [2] BIC (Bayesian Information Criterion)
#     - AIC와 유사하지만, 복잡한 모델에 더 강한 패널티 부여
#     - 데이터가 많을 때 BIC가 더 보수적인 모델을 선호함
#     - 역시 값이 작을수록 좋은 모델
#
# [3] coef (Coefficient)
#     - 각 AR, MA, Seasonal, Trend 항의 추정 계수(모델 파라미터)
#     - 계수의 부호(+, -)로 변수의 영향 방향을 해석할 수 있음
#
# [4] std err (Standard Error)
#     - 해당 계수의 추정치가 얼마나 불확실한지 나타냄
#     - 작을수록 신뢰도 높은 추정 (즉, 변동성 적음)
#
# [5] z (z-statistic)
#     - 계수 유의성 검정 통계량 (계수 / 표준오차)
#     - |z| 값이 크면, 계수가 통계적으로 유의할 가능성이 높음
#
# [6] P>|z| (p-value)
#     - 유의확률: 귀무가설(H0: 계수=0)을 기각할 확률
#     - p < 0.05 → 해당 계수가 유의하게 0이 아님 (즉, 유효한 변수)
#
# [7] Ljung-Box (Q)
#     - 잔차가 백색잡음(독립적)인지 검정
#     - p > 0.05 → 잔차에 자기상관 없음 → 모델 적합 양호
#
# [8] Jarque-Bera (JB)
#     - 잔차의 정규성(평균0, 대칭성)을 검정
#     - p > 0.05 → 정규분포 가정 만족
#
# [9] Heteroskedasticity (H)
#     - 잔차의 등분산성 여부 판단
#     - 값이 1에 가까울수록 등분산(좋음)
#
# [10] Durbin–Watson
#     - 잔차의 자기상관(특히 1차 자기상관) 여부 점검
#     - 2에 가까울수록 자기상관 없음 (1 미만: 양의 상관, 3 이상: 음의 상관)
#
# 전체적으로:
#  → AIC/BIC가 작고, 잔차 진단(Ljung-Box, JB, H, DW)이 양호하면
#     모델이 데이터 패턴을 잘 설명한다고 해석할 수 있음.
#
print(result.summary())

# ------------------------------------------------------------
# (4) 미래 12개월(1년) 예측
# ------------------------------------------------------------

# 향후 12개월 데이터를 예측 (forecast)
forecast = result.get_forecast(steps=12)

# 예측 평균값
forecast_mean = forecast.predicted_mean

# 예측 구간 (95% 신뢰구간)
forecast_ci = forecast.conf_int()

# ------------------------------------------------------------
# (5) 예측 결과 시각화
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

# 실제 관측값 (1949~1960)
plt.plot(df.index, df["Passengers"], label="Observed", color='blue')

# 예측된 평균값 (1961~1962)
plt.plot(forecast_mean.index, forecast_mean, label="Forecast", color='orange')

# 예측 신뢰구간 (Confidence Interval)
plt.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color='orange',
    alpha=0.3,
    label="Confidence Interval"
)

# 그래프 제목 및 레이블
plt.title("SARIMA Forecast for AirPassengers (1 year ahead)", fontsize=13)
plt.xlabel("Year")
plt.ylabel("Number of Passengers (in thousands)")

# 범례 및 레이아웃 조정
plt.legend()
plt.tight_layout()
plt.show()
