from entity import ModelTrainerConfig
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, classification_report
from churnPredictor import logger , CustomException
import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config = config

    def initiate_model_training(self):
        config = self.config

        # train_df = pd.read_csv(config.train_data)
        # test_df = pd.read_csv(config.test_data)

        X_train = pd.read_csv(config.train_data)
        y_train = pd.read_csv(config.y_train_path)
        X_test = pd.read_csv(config.test_data)
        y_test = pd.read_csv(config.y_test_path)
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        self.models = {
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "XGBoostClassifier": XGBClassifier(),
            "CatBoostClassifier": CatBoostClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "RandomForestClassifier": RandomForestClassifier()
        }

        trained_models = {}
        directory_path = 'artifacts/model'
        os.makedirs(directory_path, exist_ok=True)


        for model_name in self.models.keys():
            model = self.models[model_name]
            model.set_params(**dict(config.model_params_dir[model_name]))
            model.fit(X_train,y_train.values.ravel())
            logger.info(f'the {model} model trained successfully!')
            obj_name = model_name.strip('')
            obj_name = ''.join(obj_name)

            joblib.dump(model,open(file=os.path.join(r'artifacts\model',f'{obj_name}.joblib'),mode='wb'))

            trained_models[model_name] = model
        # rfc = RandomForestClassifier(n_estimators=config.n_estimators,oob_score=config.oob_score)

        # rfc.fit(X_train,y_train.values.ravel())
        # logger.info(f'the {rfc} model trained successfully')
        # joblib.dump(rfc,config.model_ojb)

        return trained_models , X_test , y_test

    def evaluate(self,true,pred):
        
        cm = confusion_matrix(true, pred)
        accuracy = accuracy_score(true, pred)
        recall = recall_score(true, pred)
        
        precision = precision_score(true, pred)
        
        report = classification_report(true, pred)

        evaluation_report = {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'classification_report': report
        }
        logger.info(f'evaluation_report -> {evaluation_report}')
        
        return evaluation_report
    
    def train_model(self):
        model ,  X_test , y_test = self.initiate_model_training()

        # y_pred = model.predict(X_test)
        # self.evaluate(y_test,y_pred)
        