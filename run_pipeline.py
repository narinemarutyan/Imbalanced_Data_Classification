import pandas as pd
import numpy as np
import argparse
import json
import joblib

from preprocessor import Preprocessor
from model import Model

parser = argparse.ArgumentParser(description='give data path and test')

parser.add_argument('--data_path', type=str, required=True, help='Enter the data path')
parser.add_argument('--inference', type=bool, default=False)

args = parser.parse_args()

path, t = args.data_path, args.inference

class Pipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = Model()
    def run(self, path, test=False):
        if test:
            x = pd.read_csv(path)
            self.preprocessor = joblib.load('preprocessor.sav')
            x = self.preprocessor.transform(x)
            self.model = joblib.load('model.sav')
            d = dict()
            d['threshold'] = 0.5
            d['predict_probas'] = self.model.predict_proba(x).tolist()
            with open('predictions.json', 'w') as outfile:
                json.dump(d, outfile)
        else:
            df = pd.read_csv(path)
            y = df['In-hospital_death']
            x = df.drop('In-hospital_death', axis=1)
            self.preprocessor.fit(df)
            x, y = self.preprocessor.transform(x, y)
            joblib.dump(self.preprocessor, 'preprocessor.sav')
            self.model.fit(x, y)
            joblib.dump(self.model, 'model.sav')


pipe = Pipeline()
pipe.run(path, t)





