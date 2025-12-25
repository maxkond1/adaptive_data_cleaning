import numpy as np
from scipy.stats import zscore

from .base import BaseCleaner
from distribution_analysis import analyze_distribution


class AdaptiveCleaner(BaseCleaner):
    def clean(self, df):
        df = df.copy()

        for col in df.columns:
            dist = analyze_distribution(df[col])

            if dist == "normal":
                df[col] = df[col].fillna(df[col].mean())
                z = np.abs(zscore(df[col]))
                df.loc[z > 3, col] = df[col].mean()

            elif dist == "skewed":
                df[col] = df[col].fillna(df[col].median())
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

            else:
                df[col] = df[col].fillna(df[col].median())

        return df
