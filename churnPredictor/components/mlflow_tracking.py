from pathlib import Path
from entity import MLFlowTrackingConfig
from churnPredictor.utils import save_json
from churnPredictor.components.model_trainer import ModelTrainer
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, classification_report
import pandas as pd
import joblib
import mlflow
from urllib.parse import urlparse
from churnPredictor import logger , CustomException


class TrackModelPerformance:
    def __init__(self,config:MLFlowTrackingConfig):
        self.config = config

    def evaluate(self,true,pred):
        
        cm = confusion_matrix(true, pred)
        accuracy = accuracy_score(true, pred)
        recall = recall_score(true, pred)
        precision = precision_score(true, pred)
        report = classification_report(true, pred)

        evaluation_report = {'confusion_matrix': cm,
                    'accuracy': accuracy,
                    'recall': recall,
                    'precision': precision,
                    'classification_report': report}
        
        logger.info(f'evaluation_report -> {evaluation_report}')
        return evaluation_report
    

    def create_experiment(self,experiment_name,run_name,model,metrics,confusion_matrix=None,params=None):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        self.tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run():

            if not params == None:
                for i in params:
                    mlflow.log_param(params,params[i])

            for metric in metrics:
                mlflow.log_metric(metric,metrics[metric])

            mlflow.sklearn.log_model(model,'Model')

            if not confusion_matrix == None:
                mlflow.log_artifact(confusion_matrix,'confusion_matrix')
            
            mlflow.set_tag("tag1", "Random Forest")
            mlflow.set_tags({"tag2":"basic model", "tag3":"experimentation"})

            logger.info('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))


    def start_mlflow(self):
        test_data = pd.read_csv(self.config.test_data)
        model = joblib.load(self.config.model_obj)
        logger.info(f'{model} loaded')
        X_test = test_data.drop('Churn',axis=1)
        y_test = test_data['Churn']

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        y_pred = model.predict(X_test)
        evaluation_report = self.evaluate(true=y_test,pred=y_pred)
        save_json(path=Path(self.config.metrics_file),data=evaluation_report)
        self.create_experiment(experiment_name='Random Forest Experiment',
                               run_name='experiment_1',
                               model=model,
                               metrics=evaluation_report,
                               params=self.config.params)

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(model, "model", registered_model_name="random forest")
        else:
            mlflow.sklearn.log_model(model, "model")
    