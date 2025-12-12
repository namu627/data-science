import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Theme once
sns.set_theme(style="whitegrid")

# Reproducibility
np.random.seed(42)
n = 150

# 공통적으로 사용할 데이터
df = pd.DataFrame({
    "category": np.random.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25]),
    "value": np.concatenate([
        np.random.normal(0.0, 1.0, 60),
        np.random.normal(0.5, 1.2, 50),
        np.random.normal(-0.3, 0.8, 40)
    ]),
    "group": np.random.choice(["G1", "G2"], size=n),
    "feature1": np.random.normal(0, 1, n),
    "feature2": np.random.normal(1, 1.2, n),
    "feature3": np.random.normal(-0.5, 0.7, n),
})

# 여기에 Seaborn 플롯 별로 사용할 코드를 여기에 붙이기
# Copy & paste codes here!
