from sklearn.impute import KNNImputer
from .base import BaseCleaner


class MultimodalCleaner(BaseCleaner):
    def clean(self, df):
        df = df.copy()
        imputer = KNNImputer(n_neighbors=5)
        df[:] = imputer.fit_transform(df)
        return df
