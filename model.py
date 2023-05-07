from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class Model:
    def __init__(self):
        self.clf = None
    def fit(self, x, y):
        self.clf = LDA().fit(x, y)
    def predict(self, x):
        return self.clf.predict(x)
    def predict_proba(self, x):
        return self.clf.predict_proba(x)


