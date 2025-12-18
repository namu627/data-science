# File: outlier_transformation_practice_class.py
# 목적: 이상치 처리 - 변환(로그, 제곱근, 표준화)을 통한 영향 완화 실습
# 환경: Python 3.10+, pandas, numpy

import numpy as np
import pandas as pd

class OutlierTransformationPractice:
    """
    이상치 처리를 위한 변환 실습 클래스
    - 로그 변환
    - 제곱근 변환
    - 표준화
    """

    def __init__(self, n: int = 500, n_outliers: int = 5, seed: int = 42):
        """
        초기화 시 데이터 생성
        :param n: 기본 표본 개수
        :param n_outliers: 극단적 이상치 개수
        :param seed: 난수 시드
        """
        self.df = self._make_income_data(n, n_outliers, seed)
        self.income = self.df["income"]

    @staticmethod
    def _make_income_data(n: int, n_outliers: int, seed: int) -> pd.DataFrame:
        """
        소득 데이터 생성 함수
        - base: 정상 소득 데이터 (로그정규 분포)
        - outliers: 비정상적으로 큰 소득 값 (극단값)
        - income: 정상 + 이상치를 합친 최종 데이터셋
        """
        rng = np.random.default_rng(seed)
        base = rng.lognormal(mean=10.5, sigma=0.6, size=n).astype(float)
        outliers = rng.lognormal(mean=13.0, sigma=0.4, size=n_outliers)
        income = np.concatenate([base, outliers])

        # ------------------------------
        # 정상값과 이상치 분포 비교 출력 (표 형태)
        # ------------------------------
        def summary(arr: np.ndarray) -> dict:
            return {
                "count": len(arr),
                "mean": np.mean(arr),
                "median": np.median(arr),
                "min": np.min(arr),
                "max": np.max(arr)
            }

        base_summary = summary(base)
        outlier_summary = summary(outliers)

        # DataFrame으로 보기 좋게 출력
        summary_df = pd.DataFrame({
            "정상값": base_summary,
            "이상치": outlier_summary
        })
        print("========== 데이터 생성 요약 ==========\n")
        # 숫자 포맷: 세자리마다 콤마, 소수점 둘째 자리까지
        formatted_df = summary_df.map(lambda x: f"{x:,.2f}")
        print(formatted_df.round(2).to_string())
        print(f"\n정상값 최대치 대비 이상치 최소치 배율: {outliers.min() / base.max():.2f}배\n")

        return pd.DataFrame({"income": income})

    # ------------------------------
    # 변환 메서드들
    # ------------------------------
    @staticmethod
    def log_transform(s: pd.Series) -> pd.Series:
        """
        로그 변환: log(x + c)
        - 로그는 0 이하 값에서 정의되지 않으므로 양의 시프트 c를 더함
        - c = 1 - min(s) + 1e-6 (min <= 0인 경우)
        """
        min_val = s.min()
        c = 0.0
        if pd.notna(min_val) and min_val <= 0:
            c = (1 - min_val) + 1e-6
        return np.log(s + c)

    @staticmethod
    def sqrt_transform(s: pd.Series) -> pd.Series:
        """
        제곱근 변환: sqrt(x + c)
        - 0 이하 값 방지를 위해 필요 시 양의 시프트 c 적용
        """
        min_val = s.min()
        c = 0.0
        if pd.notna(min_val) and min_val < 0:
            c = (-min_val) + 1e-6
        return np.sqrt(s + c)

    @staticmethod
    def standardize(s: pd.Series) -> pd.Series:
        """
        표준화: (x - μ) / σ
        - 평균을 0, 표준편차를 1로 변환
        - 이상치의 절대적 영향력이 줄어듦
        """
        mu = s.mean()
        sd = s.std(ddof=1)
        if sd == 0 or pd.isna(sd):
            return s * 0
        return (s - mu) / sd

    # ------------------------------
    # 요약 리포트
    # ------------------------------
    @staticmethod
    def _metrics(s: pd.Series) -> dict:
        """데이터 분포 요약 지표 계산"""
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        return {
            "mean": s.mean(),
            "std": s.std(ddof=1),
            "skew": s.skew(),
            "median": s.median(),
            "q1": q1,
            "q3": q3,
            "IQR": q3 - q1,
            "max": s.max()
        }

    def summary_table(self) -> pd.DataFrame:
        """
        원본 vs 변환별 분포 비교 테이블 생성
        """
        income_log = self.log_transform(self.income)
        income_sqrt = self.sqrt_transform(self.income)
        income_std = self.standardize(self.income)

        comp = pd.DataFrame({
            "원본": self._metrics(self.income),
            "로그 변환": self._metrics(income_log),
            "제곱근 변환": self._metrics(income_sqrt),
            "표준화": self._metrics(income_std)
        })
        return comp.round(4)

    # ------------------------------
    # 실행
    # ------------------------------
    def run(self) -> None:
        """실습 실행: 표 출력 및 해석 안내"""
        print("========== 이상치 변환 비교 요약 ==========\n")
        comp = self.summary_table()
        print(comp.to_string())
        print("\n해석 가이드")
        print("- 로그/제곱근 변환은 큰 값(상위 아웃라이어)의 영향력을 줄여 왜도(skew)를 완화한다.")
        print("- 표준화는 데이터의 스케일을 평균≈0, 표준편차≈1로 맞춰 이상치의 절대 크기 영향력을 줄인다.")
        print("- IQR 비교를 통해 이상치가 분포에 미치는 영향이 얼마나 줄어들었는지를 정량적으로 확인할 수 있다.")

# ------------------------------
# 메인 실행부
# ------------------------------
def main():
    practice = OutlierTransformationPractice()
    practice.run()

if __name__ == "__main__":
    main()
