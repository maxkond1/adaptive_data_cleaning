# visualization.py
# Минимальная, но информативная визуализация ДО / ПОСЛЕ очистки
# Подходит для любого числового датасета

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")


def visualize_before_after(df_before, df_after, title, max_cols=4):
    """
    Универсальная визуализация:
    - гистограммы распределений
    - boxplot для выбросов
    Ограничение по числу признаков — чтобы графиков не было слишком много
    """

    num_cols = df_before.select_dtypes(include=np.number).columns
    num_cols = num_cols[:max_cols]

    if len(num_cols) == 0:
        return

    # ---------- ГИСТОГРАММЫ ----------
    fig, axes = plt.subplots(2, len(num_cols), figsize=(4 * len(num_cols), 6))

    for i, col in enumerate(num_cols):
        sns.histplot(df_before[col], kde=True, ax=axes[0, i], color="red")
        axes[0, i].set_title(f"{col} (до)")

        sns.histplot(df_after[col], kde=True, ax=axes[1, i], color="green")
        axes[1, i].set_title(f"{col} (после)")

    plt.suptitle(f"{title}: распределения", fontsize=14)
    plt.tight_layout()
    plt.show()

    # ---------- BOXPLOT ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(data=df_before[num_cols], ax=axes[0], orient="h")
    axes[0].set_title("До очистки")

    sns.boxplot(data=df_after[num_cols], ax=axes[1], orient="h")
    axes[1].set_title("После очистки")

    plt.suptitle(f"{title}: выбросы", fontsize=14)
    plt.tight_layout()
    plt.show()
