# File: categorical_encoding_practice.py
# 목적: 범주형 변수 인코딩 비교 실습 (Label / One-Hot / Dummy)
# 환경: Python 3.10+, pandas, scikit-learn

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 1) 예제 데이터 생성
# =========================================================
def make_sample() -> pd.DataFrame:
    """
    범주형 예제 데이터 생성
    - city: 명목형(순서 없음) -> ['Seoul', 'Busan', 'Daejeon']
    - satisfaction: 순서형(서열 있음) -> ['불만족', '보통', '만족']
    - purchase: 이진 범주형 -> ['Yes', 'No']
    """
    data = {
        "city":      ["Seoul", "Busan", "Seoul", "Daejeon", "Busan", "Seoul", "Daejeon", "Seoul"],
        "satisfaction": ["만족", "보통", "만족", "불만족", "보통", "만족", "불만족", "보통"],
        "purchase":  ["Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes"],
        "amount":    [35, 12, 55, 8, 15, 48, 5, 30],  # 수치형 변수(참고용)
    }
    return pd.DataFrame(data)


# =========================================================
# 2) 헬퍼: 보기 좋게 출력
# =========================================================
def print_df(title: str, df: pd.DataFrame) -> None:
    print(f"\n========== {title} ==========\n")
    print(df.to_string(index=False))
    print()  # 줄바꿈

def print_cols(title: str, df: pd.DataFrame) -> None:
    print(f"{title}: {list(df.columns)}")


# =========================================================
# 3) 인코딩 함수들
# =========================================================
def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    라벨 인코딩
    - 문자열 범주를 정수(0,1,2,...)로 치환
    - 순서가 없는 명목형 변수에 사용하면 "숫자가 크다/작다"라는 잘못된 의미가 생길 수 있음(주의)
    """
    out = df.copy()

    # city (명목형) 라벨 인코딩: 교육용으로 보여주되, 모델에 바로 쓰기엔 부적절할 수 있음을 안내
    le_city = LabelEncoder()
    out["city_le"] = le_city.fit_transform(out["city"])

    # satisfaction (순서형) 라벨 인코딩: 순서를 반영하려면 직접 맵핑(불만족 < 보통 < 만족)
    ord_map = {"불만족": 0, "보통": 1, "만족": 2}
    out["satisfaction_le"] = out["satisfaction"].map(ord_map)

    # purchase (이진) 라벨 인코딩
    le_purchase = LabelEncoder()
    out["purchase_le"] = le_purchase.fit_transform(out["purchase"])  # Yes/No -> 1/0 (알파벳 순서)

    # 원본 범주와 매핑 정보 안내
    print("라벨 인코딩 매핑 정보")
    print(f"- city: {dict(zip(le_city.classes_, le_city.transform(le_city.classes_)))}")
    print(f"- satisfaction(순서 반영 수동 매핑): {ord_map}")
    print(f"- purchase: {dict(zip(le_purchase.classes_, le_purchase.transform(le_purchase.classes_)))}\n")

    # 라벨 인코딩 결과만 추려서 반환
    return out[["city", "city_le", "satisfaction", "satisfaction_le", "purchase", "purchase_le", "amount"]]


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    원-핫 인코딩
    - 각 범주마다 0/1 열 생성 (순서 왜곡 없음)
    - 범주 수가 많으면 열이 급격히 늘어날 수 있음(차원 증가)
    """
    # city, satisfaction, purchase에 대해 one-hot
    oh = pd.get_dummies(df, columns=["city", "satisfaction", "purchase"], dtype=int)
    return oh


def dummy_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    더미 변수 (drop_first=True)
    - 원-핫과 유사하지만 기준 범주 1개를 제거하여 다중공선성 완화
    - 회귀모형 등에서 안정성/해석성 향상
    """
    dummy = pd.get_dummies(df, columns=["city", "satisfaction", "purchase"], drop_first=True, dtype=int)
    return dummy


# =========================================================
# 4) 실행
# =========================================================
def main() -> None:
    df = make_sample()
    print_df("원본 데이터", df)

    # 라벨 인코딩
    df_label = label_encode(df)
    print_df("라벨 인코딩 결과 (LE)", df_label)

    # 원-핫 인코딩
    df_onehot = one_hot_encode(df)
    print_df("원-핫 인코딩 결과 (One-Hot)", df_onehot)
    print_cols("원-핫 인코딩 컬럼", df_onehot); print()

    # 더미 변수
    df_dummy = dummy_encode(df)
    print_df("더미 변수 결과 (Dummy, drop_first=True)", df_dummy)
    print_cols("더미 인코딩 컬럼", df_dummy); print()

    # 간단 비교 요약
    print("\n---------- 비교 요약 ---------\n")
    print(f"원본 컬럼 수: {df.shape[1]}")
    print(f"라벨 인코딩 컬럼 수: {df_label.shape[1]}")
    print(f"원-핫 인코딩 컬럼 수: {df_onehot.shape[1]}  (차원 증가 주의)")
    print(f"더미 인코딩 컬럼 수: {df_dummy.shape[1]}  (기준 범주 제거로 차원 축소)\n")

    print("\n해석 가이드")
    print("- 라벨 인코딩: 명목형에 그대로 쓰면 숫자 순서가 생겨 해석/학습 왜곡 가능.")
    print("- 원-핫 인코딩: 순서 왜곡 없음, 대신 범주 수만큼 열 증가(차원의 저주 유의).")
    print("- 더미 변수: 원-핫에서 기준 범주 1개 제거 → 다중공선성 완화, 회귀계수 해석 용이.")


if __name__ == "__main__":
    main()
