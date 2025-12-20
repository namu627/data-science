# 목적: 여러 종류의 평균(산술, 가중, 기하, 조화, 절사) 계산 예제
# 환경: Python 3.10+, numpy, scipy (선택), statistics(표준 라이브러리)

from statistics import mean, geometric_mean, harmonic_mean

import numpy as np
from scipy import stats  # trim_mean, gmean, hmean 도 제공

# 샘플 데이터: 시험 점수 또는 성과 지표 등
x = [70, 75, 80, 85, 90]

# 가중치 예시: 중간 40%, 기말 60% 같은 개념으로 비율 합이 1이 되도록
weights = [0.4, 0.0, 0.0, 0.0, 0.6]  # 단순 예시: 처음과 마지막 항목만 가중


def arithmetic_mean(arr):
    """산술평균: 모든 값을 더해 개수로 나눔. 이상치에 민감."""
    return sum(arr) / len(arr)


def weighted_mean(arr, w):
    """가중평균: 항목별 중요도(가중치)를 곱해 합한 뒤 가중치 합으로 나눔."""
    arr = np.asarray(arr, dtype=float)
    w = np.asarray(w, dtype=float)
    if np.isclose(w.sum(), 0):
        raise ValueError("가중치의 합이 0이면 안 됩니다.")
    return float((arr * w).sum() / w.sum())


def geometric_mean_safe(arr, eps=1e-12):
    """
    기하평균: 양수 데이터에 적합(성장률, 비율 등).
    0이나 음수가 포함되면 직접 계산이 불가하므로 작은 양수 eps를 더해 보정.
    보정이 필요한 상황에서는 해석에 주의.
    """
    arr = np.asarray(arr, dtype=float)
    if (arr <= 0).any():
        arr = arr + eps
    # 표준 라이브러리: statistics.geometric_mean(arr) 사용 가능
    return float(geometric_mean(arr))


def harmonic_mean_safe(arr, eps=1e-12):
    """
    조화평균: 속도, 비율처럼 '단위당' 개념에 적합. 0이나 음수에 민감.
    0이 있으면 분모가 0이라 계산 불가 → eps로 보정하되 해석에 주의.
    """
    arr = np.asarray(arr, dtype=float)
    if (arr <= 0).any():
        arr = arr + eps
    # 표준 라이브러리: statistics.harmonic_mean(arr) 사용 가능
    return float(harmonic_mean(arr))


def trimmed_mean(arr, proportion_to_cut=0.1):
    """
    절사평균: 상하위 극단값을 일정 비율만큼 잘라내고 산술평균.
    proportion_to_cut=0.1이면 상위 10%, 하위 10%씩 제거.
    이상치 영향을 완화하여 안정적인 대표값을 얻는 데 유용.
    """
    return float(stats.trim_mean(arr, proportiontocut=proportion_to_cut))


def main():
    print("========== 데이터 ==========")
    print(f"x = {x}")
    print()

    print("========== 산술평균 ==========")
    # 방법 1: 직접 구현
    print(f"산술평균(직접): {arithmetic_mean(x):.4f}")
    # 방법 2: 표준 라이브러리
    print(f"산술평균(statistics.mean): {mean(x):.4f}")
    print()

    print("========== 가중평균 ==========")
    print(f"가중치 = {weights} (합={sum(weights):.1f})")
    print(f"가중평균(직접): {weighted_mean(x, weights):.4f}")
    # numpy.average도 가중평균 지원
    print(f"가중평균(np.average): {float(np.average(x, weights=weights)):.4f}")
    print()

    print("========== 기하평균 ==========")
    # 표준 라이브러리
    print(f"기하평균(statistics.geometric_mean): {geometric_mean(x):.4f}")
    # scipy: stats.gmean(x)
    print(f"기하평균(scipy.stats.gmean): {float(stats.gmean(x)):.4f}")
    # 안전 버전(0 또는 음수 보정)
    print(f"기하평균(보정): {geometric_mean_safe(x):.4f}")
    print()

    print("========== 조화평균 ==========")
    # 표준 라이브러리
    print(f"조화평균(statistics.harmonic_mean): {harmonic_mean(x):.4f}")
    # scipy: stats.hmean(x)
    print(f"조화평균(scipy.stats.hmean): {float(stats.hmean(x)):.4f}")
    # 안전 버전(0 또는 음수 보정)
    # 데이터에 0이 포함되면 → 1/0 은 무한대가 되어서 계산 불가능
    # 데이터에 음수 값이 포함되면 → 역수가 음수로 나오고, 어떤 경우에는 전체 분모가 0에 가까워져서 비정상적인 값이 나올 수 있음
    # 안전 버전에서는 0이면 아주 작은 값(예: 1e-8)으로 치환
    print(f"조화평균(보정): {harmonic_mean_safe(x):.4f}")
    print()

    print("========== 절사평균 ==========")
    for p in (0.05, 0.1, 0.2):
        print(f"절사평균(trim {int(p*100)}%): {trimmed_mean(x, p):.4f}")
    print()

    # 참고: 극단값을 추가해 평균 간 민감도 비교
    x_with_outlier = x + [1000]
    print("========== 극단값 민감도 비교 ==========")
    print(f"원 데이터: {x} -> 산술평균 {mean(x):.2f}, 절사평균(10%) {trimmed_mean(x, 0.1):.2f}")
    print(f"이상치 추가: {x_with_outlier} -> 산술평균 {mean(x_with_outlier):.2f}, 절사평균(10%) {trimmed_mean(x_with_outlier, 0.1):.2f}")
    print()
    print("해석 가이드")
    print("- 산술평균은 이상치에 매우 민감하다.")
    print("- 절사평균은 상하위 극단값을 제거해 이상치 영향을 완화한다.")
    print("- 기하평균은 성장률/비율 데이터에, 조화평균은 속도/단위당 데이터에 적합하다.")
    print("- 가중평균은 항목 중요도가 서로 다른 경우에 사용한다.")


if __name__ == "__main__":
    main()
