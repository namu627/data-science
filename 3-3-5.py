import numpy as np
import matplotlib.pyplot as plt

# 난수 생성기 시드 고정 (재현 가능성을 위해 항상 동일한 난수 생성)
np.random.seed(19680801)  

# 데이터 딕셔너리 생성
data = {
    'a': np.arange(50),                   # 0부터 49까지 정수 (x축 데이터)
    'c': np.random.randint(0, 50, 50),    # 0~49 사이 난수 50개 (색상 값으로 활용)
    'd': np.random.randn(50)              # 표준정규분포 난수 50개 (크기 값으로 활용)
}

# y축 데이터 'b' = x축 데이터 'a' + 잡음(정규분포 * 10)
data['b'] = data['a'] + 10 * np.random.randn(50)

# 'd' 값을 양수로 변환 후 100배 확대 (산점도의 점 크기로 사용)
data['d'] = np.abs(data['d']) * 100

# Figure(전체 캔버스), Axes(데이터 영역) 생성
# figsize: 그래프 크기, layout='constrained': 자동으로 여백 최적화
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')

# 산점도(scatter plot) 그리기
# 'a' → x축, 'b' → y축, 'c' → 점 색상, 'd' → 점 크기
ax.scatter('a', 'b', c='c', s='d', data=data)

# x축, y축 라벨 설정
ax.set_xlabel('entry a')
ax.set_ylabel('entry b')

# 그래프 출력
plt.show()
