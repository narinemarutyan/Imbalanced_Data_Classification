import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class Preprocessor:
    def __init__(self):
        self.scl = None
        self.l = None
    def fit(self, df):
        self.l = list(abs(df.corr()['In-hospital_death'].values) >= 0.05)
        self.l[1] = False
        self.scl = StandardScaler()
        self.scl.fit(df.loc[:, self.l])
        self.l.pop(1)
    def transform(self, x, y=None):
        x = x.loc[:, self.l]
        x = x.fillna(x.mean())
        x = self.scl.transform(x)
        if y is None:
            return x
        x, y = SMOTE(sampling_strategy=1, random_state=42, k_neighbors=3).fit_resample(x, y)
        return x, y



