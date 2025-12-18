# File: preprocessing_deletion_demo.py
# 목적: 결측치 제거법 실습 (행 제거: listwise deletion, 열 제거: variable deletion)
# 의존성: pandas>=1.5

from __future__ import annotations
import pandas as pd
from typing import Tuple, Optional

def make_sample_dataframe() -> pd.DataFrame:
    """실습용 샘플 데이터프레임 생성"""
    data = {
        "id":      [1, 2, 3, 4, 5, 6, 7],
        "age":     [23, None, 31, 29, None, 41, 36],
        "income":  [52000, 61000, None, 58000, 60000, None, 72000],
        "city":    ["Seoul", "Busan", None, "Daejeon", "Seoul", "Seoul", None],
        "hobby":   [None, None, "Run", None, "Music", None, None],
    }
    return pd.DataFrame(data)

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼별 결측 개수와 결측률 리포트"""
    n = len(df)
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_rate":  (df.isna().mean() * 100).round(2)
    }).sort_values("missing_rate", ascending=False)
    return report

def listwise_delete(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
    keep_threshold: Optional[int] = None
) -> pd.DataFrame:
    """
    행 제거(listwise deletion)
    - subset: 지정된 컬럼들 중 하나라도 NaN이면 해당 행 삭제
    - keep_threshold(thresh): NaN이 아닌 값의 최소 개수 미만인 행을 삭제
      (예: keep_threshold=3 -> 최소 3개 이상의 유효 값이 있는 행만 유지)
    """
    if keep_threshold is not None:
        # thresh는 NaN이 아닌 값의 최소 개수 기준
        return df.dropna(thresh=keep_threshold)
    if subset is not None:
        return df.dropna(subset=subset)
    # 기본: 하나라도 NaN이 있으면 행 삭제 (모든 컬럼 기준)
    return df.dropna()

def variable_delete_by_missing_ratio(
    df: pd.DataFrame,
    threshold: float = 0.4
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    열 제거(variable deletion)
    - threshold: 결측률이 threshold 이상인 컬럼을 제거 (0~1 사이)
      예: 0.4 -> 결측률 40% 이상 컬럼 삭제
    반환: (제거 후 DF, 제거된 컬럼 목록 시리즈)
    """
    miss_rate = df.isna().mean()
    drop_cols = miss_rate[miss_rate >= threshold].index
    return df.drop(columns=drop_cols), miss_rate[drop_cols]

def main():
    # 0) 샘플 데이터 준비
    df = make_sample_dataframe()
    print("원본 데이터\n", df, "\n")
    print("결측 리포트(원본)\n", missing_report(df), "\n")

    # 1) 행 제거: 모든 컬럼 기준으로 결측 포함 행 제거 (가장 보수적)
    df_listwise_all = listwise_delete(df)
    print(f"[행 제거 - 전체 컬럼 기준] shape: {df.shape} -> {df_listwise_all.shape}")
    print(df_listwise_all, "\n")

    # 2) 행 제거: subset 기준 (예: age, income 중 하나라도 NaN이면 삭제)
    df_listwise_subset = listwise_delete(df, subset=["age", "income"])
    print(f"[행 제거 - subset=['age','income']] shape: {df.shape} -> {df_listwise_subset.shape}")
    print(df_listwise_subset, "\n")

    # 3) 행 제거: keep_threshold 사용 (유효값이 4개 미만이면 삭제)
    df_listwise_thresh = listwise_delete(df, keep_threshold=4)
    print(f"[행 제거 - keep_threshold=4] shape: {df.shape} -> {df_listwise_thresh.shape}")
    print(df_listwise_thresh, "\n")

    # 4) 열 제거: 결측률 40% 이상인 컬럼 제거
    df_col_drop_40, dropped_40 = variable_delete_by_missing_ratio(df, threshold=0.40)
    print("[열 제거 - 결측률 40% 이상 삭제] 제거된 컬럼:")
    print(dropped_40.apply(lambda r: f"{round(r*100,2)}%"), "\n")
    print(f"shape: {df.shape} -> {df_col_drop_40.shape}")
    print(df_col_drop_40, "\n")
    print("결측 리포트(열 제거 후)\n", missing_report(df_col_drop_40), "\n")

    # 5) 열 제거: 결측률 60% 이상인 컬럼만 더 강하게 제거
    df_col_drop_60, dropped_60 = variable_delete_by_missing_ratio(df, threshold=0.60)
    print("[열 제거 - 결측률 60% 이상 삭제] 제거된 컬럼:")
    print(dropped_60.apply(lambda r: f"{round(r*100,2)}%"), "\n")
    print(f"shape: {df.shape} -> {df_col_drop_60.shape}")
    print(df_col_drop_60, "\n")

    # 6) 파이프라인 예시: 먼저 결측률 높은 열 제거 -> 그 다음 subset 기준 행 제거
    df_pipe, _ = variable_delete_by_missing_ratio(df, threshold=0.5)
    df_pipe = listwise_delete(df_pipe, subset=["age", "income"])
    print("[파이프라인] (열 제거 50%) -> (subset 행 제거)")
    print(f"최종 shape: {df.shape} -> {df_pipe.shape}")
    print(df_pipe, "\n")

    # 팁: 실제 분석에서는 제거 전후로 분포가 어떻게 달라지는지 반드시 확인
    # 예) df['age'].describe(), df_listwise_subset['age'].describe() 등 비교

if __name__ == "__main__":
    main()
