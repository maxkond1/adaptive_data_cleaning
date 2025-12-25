from .base import BaseCleaner


class SkewedDistributionCleaner(BaseCleaner):
    def clean(self, df):
        df = df.copy()
        for col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return df
