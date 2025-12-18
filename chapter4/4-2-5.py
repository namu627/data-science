# File: preprocessing_imputation_demo_clean_min.py
# 목적: 결측치 대체(Imputation) 기본 실습
# 기능: 단순 대체(평균·최빈값), KNN 대체, 회귀 기반 대체
# 설명 수준: 데이터사이언스 입문 교재용 (친절한 주석 포함)

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # IterativeImputer 사용 가능하게 함
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


# =========================
# 데이터 준비
# =========================
def make_sample_dataframe() -> pd.DataFrame:
    """실습용 샘플 데이터프레임 생성"""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7],
        "age": [23, None, 31, 29, None, 41, 36],
        "income": [52000, 61000, None, 58000, 60000, None, 72000],
        "city": ["Seoul", "Busan", None, "Daejeon", "Seoul", "Seoul", None],
        "hobby": [None, None, "Run", None, "Music", None, None],
    })


# =========================
# 유틸리티 함수
# =========================
def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """None → np.nan 변환"""
    return df.replace({None: np.nan})


def detect_dtypes(df: pd.DataFrame, exclude: list[str] | None = None) -> tuple[list[str], list[str]]:
    """수치형/범주형 컬럼 구분"""
    exclude = exclude or []
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in exclude]
    return num_cols, cat_cols


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """결측 개수와 결측률 리포트"""
    return pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_rate": (df.isna().mean() * 100).round(2)
    }).sort_values("missing_rate", ascending=False)


# =========================
# 결측치 대체 방법
# =========================
def simple_impute_mean_mode(df: pd.DataFrame) -> pd.DataFrame:
    """단순 대체: 수치형=평균, 범주형=최빈값"""
    df = normalize_missing(df.copy())
    num_cols, cat_cols = detect_dtypes(df, exclude=["id"])
    if num_cols:
        df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    return df


def knn_impute(df: pd.DataFrame, n_neighbors: int = 3) -> pd.DataFrame:
    """KNN 기반 대체"""
    df = normalize_missing(df.copy())
    num_cols, cat_cols = detect_dtypes(df, exclude=["id"])
    out = df.copy()
    if num_cols:
        out[num_cols] = KNNImputer(n_neighbors=n_neighbors).fit_transform(df[num_cols])
    if cat_cols:
        out[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    return out


def iterative_impute(df: pd.DataFrame) -> pd.DataFrame:
    """회귀 기반 대체 (Iterative Imputer)"""
    df = normalize_missing(df.copy())
    num_cols, cat_cols = detect_dtypes(df, exclude=["id"])
    out = df.copy()
    if num_cols:
        it = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=42,
            sample_posterior=True
        )
        out[num_cols] = it.fit_transform(df[num_cols])
    if cat_cols:
        out[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(out[cat_cols])
    return out


# =========================
# 실행 예제
# =========================
def main() -> None:
    df = make_sample_dataframe()

    print("========== 원본 데이터 ==========\n")
    print(f"{df}\n")
    print("결측 리포트(원본)\n")
    print(f"{missing_report(df)}\n")

    print("---------- 단순 대체 (평균/최빈값) ---------\n")
    df1 = simple_impute_mean_mode(df)
    print(f"{df1}\n")
    print("결측 리포트(단순 대체)\n")
    print(f"{missing_report(df1)}\n")

    print("---------- KNN 기반 대체 ---------\n")
    df2 = knn_impute(df, n_neighbors=3)
    print(f"{df2}\n")
    print("결측 리포트(KNN)\n")
    print(f"{missing_report(df2)}\n")

    print("---------- 회귀 기반 대체 (Iterative) ---------\n")
    df3 = iterative_impute(df)
    print(f"{df3.round(2)}\n")  # 보기 좋게 소수점 반올림
    print("결측 리포트(Iterative)\n")
    print(f"{missing_report(df3)}\n")


if __name__ == "__main__":
    main()
