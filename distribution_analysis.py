import numpy as np
from scipy.stats import skew, shapiro


def analyze_distribution(series):
    s = series.dropna()

    if len(s) < 30:
        return "small_sample"

    sk = skew(s)

    try:
        p = shapiro(s.sample(5000) if len(s) > 5000 else s)[1]
    except:
        p = 0

    if abs(sk) < 0.5 and p > 0.05:
        return "normal"
    elif abs(sk) > 1:
        return "skewed"
    else:
        return "multimodal"
