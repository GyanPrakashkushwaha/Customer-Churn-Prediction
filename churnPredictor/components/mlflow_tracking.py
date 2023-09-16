from churnPredictor.entity import MLFlowTrackingConfig
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, classification_report
import pandas as pd
import joblib
import mlflow
from churnPredictor import logger , CustomException
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

class TrackModelPerformance:
    def __init__(self,config:MLFlowTrackingConfig):
        self.config = config

    def evaluate(self,true,pred):
        
        cm = confusion_matrix(true, pred)
        sns.heatmap(data=cm,annot=True, fmt='d', cmap='Blues')
        plt.savefig(self.config.confusion_metrics)
        accuracy = accuracy_score(true, pred)
        recall = recall_score(true, pred)
        precision = precision_score(true, pred)
        report = classification_report(true, pred)

        evaluation_report = {
                    'accuracy': accuracy,
                    'recall': recall,
                    'precision': precision,
                    # 'classification_report': report
                    }
        
        logger.info(f'evaluation_report -> {evaluation_report}')
        return evaluation_report 
    

    def start_mlflow(self):
        try:
            # I will commit this to use MLFlow offline and run ```mlflow ui``` and for online everything remain uncommented
            
            mlflow.set_tracking_uri('https://dagshub.com/GyanPrakashKushwaha/Customer-Churn-Prediction.mlflow')
            os.environ["MLFLOW_TRACKING_USERNAME"]="GyanPrakashKushwaha"
            os.environ["MLFLOW_TRACKING_PASSWORD"]= '53950624aa84e08b2bd1dfb3c0778ff66c4e7d05'


            X_test = pd.read_csv(self.config.test_data)
            
            models = {
                    "GradientBoostingClassifier": GradientBoostingClassifier(),
                    "XGBoostClassifier": XGBClassifier(),
                    "CatBoostClassifier": CatBoostClassifier(),
                    "AdaBoostClassifier": AdaBoostClassifier(),
                    "RandomForestClassifier": RandomForestClassifier()
                }

            for model_name in models.keys():
                model = joblib.load(open(file=os.path.join(r'artifacts\model',f'{model_name}.joblib'),mode='rb'))
            
                logger.info(f'{model} loaded')
                # X_test = test_data.drop('Churn',axis=1)

                y_test = pd.read_csv(self.config.y_test_path)
                mlflow.set_experiment(model_name)

                with mlflow.start_run():
                    y_pred = model.predict(X_test)
                    evaluation_report = self.evaluate(true=y_test,pred=y_pred)
                    with open(self.config.metrics_file, 'w') as json_file:
                        json.dump(evaluation_report, json_file)
                    if not self.config.params == None:
                        for param in self.config.params:
                            mlflow.log_param(param, self.config.params[param])
                    # mlflow.log_params(self.config.params)

                    for metric in evaluation_report:
                        mlflow.log_metric(metric,evaluation_report[metric])
                                    
                # if tracking_url_type_store != 'file':
                #     mlflow.sklearn.log_model(model, 'model', registered_model_name="random forest")
                # else:
                    mlflow.sklearn.log_model(model, self.config.model_obj)
        except Exception as e:
            raise CustomException(e)
