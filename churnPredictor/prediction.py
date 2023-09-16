import joblib
import numpy 
import pandas


class Prediction:
    def __init__(self) :
        self.model = joblib.load(open(r'artifacts\model\model.joblib','rb'))

    def predict(self,test_data):
        preds = self.model.predict(test_data)
        return preds
