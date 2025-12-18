# File: normalization_practice_class.py
# 목적: 정규화/스케일링 실습 (Min-Max, Z-score, Robust)
# 환경: Python 3.10+, pandas, numpy, scikit-learn

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class NormalizationPractice:
    """
    정규화/스케일링 실습 클래스
    - 데이터: height_cm(키), weight_kg(몸무게), income_m(소득/만원)
    - 스케일러: Min-Max, Z-score(Standard), Robust(IQR 기반)
    - 출력: 변수별로 원본 vs 변환 후 요약 지표 비교 표
    """

    def __init__(self, n: int = 300, n_outliers: int = 5, seed: int = 42):
        self.df = self._make_dataset(n, n_outliers, seed)
        self.num_cols = ["height_cm", "weight_kg", "income_m"]

    @staticmethod
    def _make_dataset(n: int, n_outliers: int, seed: int) -> pd.DataFrame:
        """
        서로 다른 스케일의 변수를 가지는 데이터 생성
        - height_cm: N(170, 7^2)
        - weight_kg: N(70, 12^2)
        - income_m (만원): 로그정규 분포(긴 꼬리) + 상위 아웃라이어 몇 개
        """
        rng = np.random.default_rng(seed)

        # 키(cm): 평균 170, 표준편차 7
        height = rng.normal(loc=170, scale=7, size=n)

        # 몸무게(kg): 평균 70, 표준편차 12
        weight = rng.normal(loc=70, scale=12, size=n)

        # 소득(만원): 로그정규로 긴 꼬리 분포, 스케일 차이를 크게 만들기 위해 100배 스케일링
        income_base = rng.lognormal(mean=2.8, sigma=0.5, size=n) * 100

        # 상위 아웃라이어 추가(소득만): 현실적인 긴 꼬리를 더 강조
        income_out = rng.lognormal(mean=4.2, sigma=0.25, size=n_outliers) * 100
        income = np.concatenate([income_base, income_out])

        # 길이를 맞추기 위해 height/weight에도 동일 개수만큼 샘플 추가
        height_out = rng.normal(loc=170, scale=7, size=n_outliers)
        weight_out = rng.normal(loc=70, scale=12, size=n_outliers)

        height = np.concatenate([height, height_out])
        weight = np.concatenate([weight, weight_out])

        df = pd.DataFrame(
            {
                "height_cm": height.astype(float),
                "weight_kg": weight.astype(float),
                "income_m": income.astype(float),  # 만원 단위
            }
        )
        return df

    # ------------------------------
    # 스케일러 적용
    # ------------------------------
    def _fit_transform(self, scaler) -> pd.DataFrame:
        """주어진 스케일러로 수치 컬럼 변환"""
        arr = scaler.fit_transform(self.df[self.num_cols])
        out = pd.DataFrame(arr, columns=self.num_cols, index=self.df.index)
        return out

    def minmax(self) -> pd.DataFrame:
        return self._fit_transform(MinMaxScaler())

    def zscore(self) -> pd.DataFrame:
        return self._fit_transform(StandardScaler())

    def robust(self) -> pd.DataFrame:
        return self._fit_transform(RobustScaler(with_centering=True, with_scaling=True))

    # ------------------------------
    # 요약 지표
    # ------------------------------
    @staticmethod
    def _metrics(s: pd.Series) -> dict:
        """분포 요약 지표: 평균, 표준편차, 최솟값, Q1, 중앙값, Q3, IQR, 최댓값"""
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        return {
            "mean": s.mean(),
            "std": s.std(ddof=1),
            "min": s.min(),
            "q1": q1,
            "median": s.median(),
            "q3": q3,
            "IQR": q3 - q1,
            "max": s.max(),
        }

    def _summary_by_column(self, col: str, df_minmax: pd.DataFrame,
                           df_z: pd.DataFrame, df_robust: pd.DataFrame) -> pd.DataFrame:
        """
        단일 변수에 대해 원본/세 스케일러 결과를 한 표로 정리
        행: 지표, 열: 원본 | Min-Max | Z-score | Robust
        """
        original = self._metrics(self.df[col])
        mmin = self._metrics(df_minmax[col])
        zsc = self._metrics(df_z[col])
        rbs = self._metrics(df_robust[col])

        tbl = pd.DataFrame(
            {
                "원본": original,
                "Min-Max": mmin,
                "Z-score": zsc,
                "Robust": rbs,
            }
        ).round(4)

        return tbl

    # ------------------------------
    # 실행
    # ------------------------------
    def run(self) -> None:
        # 스케일러 적용
        df_minmax = self.minmax()
        df_z = self.zscore()
        df_rb = self.robust()

        # 원본 범위 미리보기(단위 차이 확인)
        print("========== 원본 데이터 범위 미리보기 ==========\n")
        preview = pd.DataFrame(
            {
                "min": self.df[self.num_cols].min(),
                "median": self.df[self.num_cols].median(),
                "max": self.df[self.num_cols].max(),
            }
        )
        # 소수점 2자리 및 세자리 콤마 포맷
        print(preview.map(lambda x: f"{x:,.2f}").to_string())
        print()

        # 변수별 요약 표 출력
        for col in self.num_cols:
            print(f"---------- 변수: {col} ----------\n")
            tbl = self._summary_by_column(col, df_minmax, df_z, df_rb)

            # 보기 좋게 포맷: std/IQR 제외한 값은 콤마, 소수점 2자리
            def fmt(x):
                try:
                    return f"{x:,.2f}"
                except Exception:
                    return x

            formatted = tbl.copy()
            for c in formatted.columns:
                formatted[c] = formatted[c].apply(fmt)

            print(formatted.to_string())
            print()

        # 해석 가이드
        print("해석 가이드")
        print("- Min-Max: 각 변수의 min→0, max→1. 범위가 0~1로 통일되어 거리 기반 모델에 유리.")
        print("- Z-score: mean≈0, std≈1로 맞춰 경사 하강법 기반 모델 학습 안정화에 유리.")
        print("- Robust: 중앙값 기준, IQR 스케일(이상치에 강건). 이상치 많은 데이터에 유리.")


# ------------------------------
# 메인 실행부
# ------------------------------
def main():
    app = NormalizationPractice(n=300, n_outliers=5, seed=42)
    app.run()


if __name__ == "__main__":
    main()
