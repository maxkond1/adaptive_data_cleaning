import numpy as np
from scipy.stats import zscore
from .base import BaseCleaner


class NormalDistributionCleaner(BaseCleaner):
    def clean(self, df):
        df = df.copy()
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
            z = np.abs(zscore(df[col]))
            df.loc[z > 3, col] = df[col].mean()
        return df
