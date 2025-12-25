from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from .base import BaseCleaner
import numpy as np
import pandas as pd

class HeavyOutliersCleaner(BaseCleaner):
    def clean(self, df):
        df = df.copy()

        # 1. Разделяем числовые и категориальные признаки
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns

        # 2. Импутация пропусков
        if len(num_cols) > 0:
            df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
        if len(cat_cols) > 0:
            df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

        # 3. Обнаружение и корректировка выбросов в числовых
        if len(num_cols) > 0:
            iso = IsolationForest(contamination=0.05, random_state=42)
            flags = iso.fit_predict(df[num_cols])
            outlier_idx = np.where(flags == -1)[0]
            median_values = df[num_cols].median().values
            df.iloc[outlier_idx, df.columns.get_indexer(num_cols)] = median_values
            print(f"Аномалии исправлено: {len(outlier_idx)}")

        return df
