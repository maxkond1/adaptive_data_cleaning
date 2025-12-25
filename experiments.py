# experiments.py
from datasets_loader import load_all_datasets
from cleaners import (
    NormalDistributionCleaner,
    SkewedDistributionCleaner,
    MultimodalCleaner,
    HeavyOutliersCleaner,
    AdaptiveCleaner
)
from visualization import visualize_before_after

# ===============================
# Простое правило выбора алгоритма
# ===============================
def select_cleaner(df):
    # Простейшее правило на основе типа данных и скошенности
    num_cols = df.select_dtypes(include="number")
    if num_cols.empty:
        return AdaptiveCleaner()  # если нет числовых признаков
    skew_values = num_cols.skew().abs()
    max_skew = skew_values.max()

    if max_skew < 0.5:
        return NormalDistributionCleaner()
    elif max_skew < 2:
        return SkewedDistributionCleaner()
    elif max_skew >= 2:
        return HeavyOutliersCleaner()
    else:
        return AdaptiveCleaner()


def run_experiment(name, df):
    print(f"\n===== {name} =====")
    df_before = df.copy()
    cleaner = select_cleaner(df)
    df_after = cleaner.clean(df)

    print("Пропусков до:", df_before.isnull().sum().sum())
    print("Пропусков после:", df_after.isnull().sum().sum())

    visualize_before_after(df_before, df_after, name)


def run_all():
    datasets = load_all_datasets()
    for name, df in datasets.items():
        run_experiment(name, df)


if __name__ == "__main__":
    run_all()
