import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. 데이터 준비
# -----------------------------

# 난수 시드를 고정하여 실행할 때마다 동일한 결과가 나오도록 설정
np.random.seed(19680801)

# 정규분포를 따르는 난수 100개를 누적합(cumulative sum)하여 시계열 데이터 생성
data1 = np.random.randn(100).cumsum()
data2 = np.random.randn(100).cumsum()

# -----------------------------
# 2. Figure와 Axes 생성
# -----------------------------

# 5x2.7 인치 크기의 Figure와 하나의 Axes 생성
fig, ax = plt.subplots(figsize=(5, 2.7))

# x축 값: 0부터 데이터 길이(100)까지 정수 배열
x = np.arange(len(data1))

# -----------------------------
# 3. 선 그래프 그리기
# -----------------------------

# 첫 번째 선: data1 값 시각화
# 색상: 파란색, 선 굵기: 3, 선 스타일: '--' (점선)
ax.plot(x, data1, color='blue', linewidth=3, linestyle='--')

# 두 번째 선: data2 값 시각화
# 색상: 주황색, 선 굵기: 2
# 반환값 l은 Line2D 객체로, 이후 스타일 변경에 사용됨
l, = ax.plot(x, data2, color='orange', linewidth=2)

# -----------------------------
# 4. 스타일 사후 변경
# -----------------------------

# 두 번째 선의 선 스타일을 ':' (점선)으로 변경
l.set_linestyle(':')

# -----------------------------
# 5. 그래프 출력
# -----------------------------

plt.show()
