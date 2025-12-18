# File: domain_imputation_practice_class.py
# 목적: 도메인 지식 활용 실습 (클래스 기반)
# 내용: 0과 미기록(NaN)을 구분해 결측을 처리하는 방법 비교
# 환경: Python 3.10+, pandas, numpy, scikit-learn

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class DomainImputationPractice:
    """
    도메인 지식 활용 결측치 처리 실습을 클래스 형태로 구성
    - make_dataset: 샘플 환자 데이터 생성
    - normalize_missing: None → np.nan 통일
    - missing_report / numeric_summary / categorical_counts: 리포트·요약
    - naive_impute: 접근 A (0을 값으로 간주, NaN만 대체)
    - domain_mark_missing / domain_impute: 접근 B (0을 결측으로 재분류 후 대체)
    - run: 전체 시나리오 실행 및 비교 출력
    """

    def __init__(self):
        # 분석 대상 컬럼 정의
        # 연속형 생체지표: 0은 생리학적으로 불가능 → 도메인 판단 시 결측으로 재분류
        self.numeric_cols: list[str] = ["sbp", "dbp", "glucose"]
        # 이진 변수: 0은 '증상 없음'이라는 실제 값 → 유지
        self.binary_cols: list[str] = ["symptom_present"]
        # 범주형 변수
        self.cat_cols: list[str] = ["diagnosis"]

        # 0을 결측으로 재분류할 컬럼 목록(도메인 규칙)
        self.zero_is_missing: list[str] = ["sbp", "dbp", "glucose"]

    # =========================
    # 데이터 생성 및 유틸
    # =========================
    def make_dataset(self) -> pd.DataFrame:
        """
        환자 데이터 샘플 생성
        - sbp, dbp, glucose는 연속형 생체지표 (0과 NaN 섞어 둠)
        - symptom_present는 0/1 이진 변수 (0은 의미 있는 실제 값)
        - diagnosis는 범주형 (None=미기록, "None" 문자열은 실제 '진단 없음')
        """
        df = pd.DataFrame(
            {
                "patient_id": [101, 102, 103, 104, 105, 106, 107, 108],
                "sbp":        [120, 0,   138, 145, np.nan, 132, 0,   118],
                "dbp":        [80,  0,   92,  95,  88,     np.nan, 0,   76],
                "glucose":    [98,  0,   np.nan, 115, 102, 0,     140, np.nan],
                "symptom_present": [0, 1, 0, 1, 0, 0, 1, np.nan],
                "diagnosis":  ["Hypertension", None, "Diabetes", "Hypertension",
                               "None", None, "Diabetes", "None"],
            }
        )
        return df

    @staticmethod
    def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
        """
        None → np.nan 통일
        - pandas/Scikit-Learn에서 결측을 일관되게 다루기 위함
        """
        return df.replace({None: np.nan})

    @staticmethod
    def missing_report(df: pd.DataFrame) -> pd.DataFrame:
        """
        각 컬럼별 결측 개수와 결측률 출력용 표 생성
        """
        return pd.DataFrame(
            {
                "missing_count": df.isna().sum(),
                "missing_rate": (df.isna().mean() * 100).round(2),
            }
        ).sort_values("missing_rate", ascending=False)

    @staticmethod
    def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
        수치형 요약 통계 반환
        """
        if not cols:
            return pd.DataFrame()
        return df[cols].describe().T

    @staticmethod
    def categorical_counts(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.Series]:
        """
        범주형 분포 반환
        - dropna=False로 결측까지 카운트
        """
        out: dict[str, pd.Series] = {}
        for c in cols:
            out[c] = df[c].value_counts(dropna=False)
        return out

    @staticmethod
    def print_cat_counts(title: str, counts: dict[str, pd.Series]) -> None:
        print(title)
        if not counts:
            print("(범주형 컬럼 없음)\n")
            return
        for k, s in counts.items():
            print(f"[{k}]")
            print(f"{s}\n")

    # =========================
    # 접근 A: 단순 방식
    # =========================
    def naive_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        단순 방식
        - 0을 그대로 '유효한 값'으로 간주
        - NaN만 대체
          - 연속형(sbp, dbp, glucose): 중앙값
          - 이진/범주형(symptom_present, diagnosis): 최빈값
        """
        work = self.normalize_missing(df.copy())

        if self.numeric_cols:
            work[self.numeric_cols] = SimpleImputer(strategy="median").fit_transform(work[self.numeric_cols])
        if self.binary_cols:
            work[self.binary_cols] = SimpleImputer(strategy="most_frequent").fit_transform(work[self.binary_cols])
        if self.cat_cols:
            work[self.cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(work[self.cat_cols])

        return work

    # =========================
    # 접근 B: 도메인 방식
    # =========================
    def domain_mark_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        도메인 규칙으로 '0을 미기록으로 간주해야 하는 컬럼'을 NaN으로 치환
        - sbp, dbp, glucose: 0은 생리학적으로 불가능 → 결측으로 재분류
        - symptom_present: 0은 '증상 없음'이라는 실제 값 → 유지
        - diagnosis: "None" 문자열은 실제 값 → 유지, NaN은 미기록
        """
        work = self.normalize_missing(df.copy())

        for col in self.zero_is_missing:
            if col in work.columns:
                work.loc[work[col] == 0, col] = np.nan

        return work

    def domain_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        도메인 방식
        - 0을 결측으로 재분류한 뒤 impute
          - 연속형: 중앙값
          - 이진/범주형: 최빈값
        """
        work = self.domain_mark_missing(df)

        if self.numeric_cols:
            work[self.numeric_cols] = SimpleImputer(strategy="median").fit_transform(work[self.numeric_cols])
        if self.binary_cols:
            work[self.binary_cols] = SimpleImputer(strategy="most_frequent").fit_transform(work[self.binary_cols])
        if self.cat_cols:
            work[self.cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(work[self.cat_cols])

        return work

    # =========================
    # 실행 흐름
    # =========================
    def run(self) -> None:
        # 데이터 로드
        df = self.make_dataset()

        print("========== 원본 데이터 ==========\n")
        print(f"{df}\n")
        print("결측 리포트(원본)\n")
        print(f"{self.missing_report(df)}\n")
        print("수치형 요약 통계(원본)\n")
        print(f"{self.numeric_summary(df, self.numeric_cols)}\n")
        self.print_cat_counts("범주형 분포(원본)\n", self.categorical_counts(df, self.binary_cols + self.cat_cols))

        # 접근 A: 단순 방식
        print("---------- 접근 A: 단순 방식 (0을 값으로 간주, NaN만 대체) ---------\n")
        naive = self.naive_impute(df)
        print(f"{naive}\n")
        print("결측 리포트(접근 A)\n")
        print(f"{self.missing_report(naive)}\n")
        print("수치형 요약 통계(접근 A)\n")
        print(f"{self.numeric_summary(naive, self.numeric_cols)}\n")
        self.print_cat_counts("범주형 분포(접근 A)\n", self.categorical_counts(naive, self.binary_cols + self.cat_cols))

        # 접근 B: 도메인 방식
        print("---------- 접근 B: 도메인 방식 (0을 결측으로 재분류 후 대체) ---------\n")
        domain_pre = self.domain_mark_missing(df)
        # 0 → NaN 재분류 건수 확인
        zero_to_nan_counts = {
            c: int(((df[c] == 0) if c in df.columns else pd.Series([], dtype=bool)).sum())
            for c in self.zero_is_missing
        }
        print(f"0을 결측으로 재분류한 건수: {zero_to_nan_counts}\n")
        print("0 재분류 적용 후 데이터 스냅샷(미대체)\n")
        print(f"{domain_pre}\n")
        print("결측 리포트(도메인 재분류 후, 미대체)\n")
        print(f"{self.missing_report(domain_pre)}\n")

        domain = self.domain_impute(df)
        print("도메인 방식 대체 완료 데이터\n")
        print(f"{domain}\n")
        print("결측 리포트(접근 B)\n")
        print(f"{self.missing_report(domain)}\n")
        print("수치형 요약 통계(접근 B)\n")
        print(f"{self.numeric_summary(domain, self.numeric_cols)}\n")
        self.print_cat_counts("범주형 분포(접근 B)\n", self.categorical_counts(domain, self.binary_cols + self.cat_cols))

        # 간단 비교 요약
        print("========== 간단 비교 요약 ==========\n")
        mean_comp = pd.DataFrame(
            {
                "원본": df[self.numeric_cols].mean(numeric_only=True),
                "접근A(단순)": naive[self.numeric_cols].mean(numeric_only=True),
                "접근B(도메인)": domain[self.numeric_cols].mean(numeric_only=True),
            }
        ).round(2)
        print("수치형 평균 비교\n")
        print(f"{mean_comp}\n")

        median_comp = pd.DataFrame(
            {
                "원본": df[self.numeric_cols].median(numeric_only=True),
                "접근A(단순)": naive[self.numeric_cols].median(numeric_only=True),
                "접근B(도메인)": domain[self.numeric_cols].median(numeric_only=True),
            }
        ).round(2)
        print("수치형 중앙값 비교\n")
        print(f"{median_comp}\n")

        print("이진/범주형 분포 비교\n")
        for col in self.binary_cols + self.cat_cols:
            orig_counts = df[col].value_counts(dropna=False)
            a_counts = naive[col].value_counts(dropna=False)
            b_counts = domain[col].value_counts(dropna=False)
            print(f"[{col}]")
            print(f"  원본 분포:\n{orig_counts}\n")
            print(f"  접근A 분포:\n{a_counts}\n")
            print(f"  접근B 분포:\n{b_counts}\n")

def main():
    app = DomainImputationPractice()
    app.run()


if __name__ == "__main__":
    main()
