from pathlib import Path
from churnPredictor.entity import MLFlowTrackingConfig
from churnPredictor.utils import save_json
from churnPredictor.components.model_trainer import ModelTrainer
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, classification_report
import pandas as pd
import joblib
import mlflow
from urllib.parse import urlparse
from churnPredictor import logger , CustomException
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
    

    def create_experiment(self,experiment_name,run_name,model,metrics,confusion_matrix=None,params=None):
        try:
            mlflow.set_tracking_uri('https://dagshub.com/GyanPrakashKushwaha/Customer-Churn-Prediction.mlflow')
            os.environ["MLFLOW_TRACKING_USERNAME"]="GyanPrakashKushwaha"
            os.environ["MLFLOW_TRACKING_PASSWORD"]= '53950624aa84e08b2bd1dfb3c0778ff66c4e7d05'
            
            # mlflow.
            mlflow.set_registry_uri(self.config.mlflow_uri)
            self.tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            mlflow.set_experiment(experiment_name=experiment_name)
            with mlflow.start_run():

                if not params == None:
                    for i in params:
                        mlflow.log_param(i,params[i])

                for metric in metrics:
                    mlflow.log_metric(metric,metrics[metric])


                if not confusion_matrix == None:
                    mlflow.log_artifact(confusion_matrix,'confusion_matrix')
                
                # mlflow.log_metric('mse',323)
                mlflow.log_param('tree',100)
                mlflow.sklearn.log_model(model,self.config.model_obj)
                
                mlflow.set_tag("tag1", "Random Forest")
                mlflow.set_tags({"tag2":"basic model", "tag3":"experimentation"})

                logger.info('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))
        except Exception as e:
            raise CustomException(e)


    def start_mlflow(self):
        try:
            test_data = pd.read_csv(self.config.test_data)
            model = pickle.load(open(self.config.model_obj,'rb'))
            logger.info(f'{model} loaded')
            X_test = test_data.drop('Churn',axis=1)
            y_test = test_data['Churn']

            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            y_pred = model.predict(X_test)
            evaluation_report = self.evaluate(true=y_test,pred=y_pred)
            with open(self.config.metrics_file, 'w') as json_file:
                json.dump(evaluation_report, json_file)

            
            self.create_experiment(experiment_name='Random-Forest',
                                run_name='experiment_1',
                                model=model,
                                metrics=evaluation_report,
                                params=self.config.params,
                                confusion_matrix=self.config.confusion_metrics)

            if tracking_url_type_store != 'file':
                mlflow.sklearn.log_model(model, self.config.model_obj, registered_model_name="random forest")
            else:
                mlflow.sklearn.log_model(model,  self.config.model_obj, registered_model_name="random forest")
        except Exception as e:
            raise CustomException(e)
