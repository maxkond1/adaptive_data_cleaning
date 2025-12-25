# datasets_loader.py
# Универсальная загрузка рекомендованного набора реальных датасетов
# БЕЗ Kaggle API, БЕЗ токенов — всё загружается автоматически по URL

import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing,
    load_breast_cancer
)


# =========================
# 1. СИНТЕТИЧЕСКИЕ ДАТАСЕТЫ
# =========================

def load_synthetic_normal():
    np.random.seed(42)
    data = np.random.normal(50, 10, size=(1000, 5))
    df = pd.DataFrame(data, columns=[f"N{i}" for i in range(5)])
    df.iloc[::10, 0] = np.nan
    df.iloc[::15, 1] *= 5
    return df


def load_synthetic_skewed():
    np.random.seed(42)
    data = np.random.exponential(scale=2, size=(1000, 4))
    df = pd.DataFrame(data, columns=[f"S{i}" for i in range(4)])
    df.iloc[::12, 0] = np.nan
    return df


# =========================
# 2. SCIKIT-LEARN DATASETS
# =========================

def load_diabetes_dataset():
    return load_diabetes(as_frame=True).frame


def load_california_housing():
    return fetch_california_housing(as_frame=True).frame


def load_breast_cancer_dataset():
    return load_breast_cancer(as_frame=True).frame


# =========================
# 3. REAL CSV DATASETS (UCI / Kaggle mirrors)
# =========================

def load_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)


def load_house_prices():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    return pd.read_csv(url)


def load_wine_quality():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    return pd.read_csv(url, sep=";")


def load_adult_income():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    return pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)


# =========================
# 4. ЕДИНАЯ ТОЧКА ДОСТУПА
# =========================

def load_all_datasets():
    return {
        "synthetic_normal": load_synthetic_normal(),
        "synthetic_skewed": load_synthetic_skewed(),
        "diabetes": load_diabetes_dataset(),
        "california_housing": load_california_housing(),
        "breast_cancer": load_breast_cancer_dataset(),
        "titanic": load_titanic(),
        "house_prices": load_house_prices(),
        "wine_quality": load_wine_quality(),
        "adult_income": load_adult_income(),
    }


# =========================
# 5. ТЕСТ
# =========================

if __name__ == "__main__":
    datasets = load_all_datasets()
    for name, df in datasets.items():
        print(f"{name}: {df.shape}")
