from entity import ModelTrainerConfig
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, classification_report
from churnPredictor import logger , CustomException
import joblib
import pickle



class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config = config

    def initiate_model_training(self):
        config = self.config

        train_df = pd.read_csv(config.train_data)
        test_df = pd.read_csv(config.test_data)

        X_train = train_df.drop('Churn',axis=1)
        y_train = train_df['Churn']
        X_test = test_df.drop('Churn',axis=1)
        y_test = test_df['Churn']

        rfc = RandomForestClassifier(n_estimators=config.n_estimators,oob_score=config.oob_score)


        rfc.fit(X_train,y_train)
        logger.info(f'the {rfc} model trained successfully')
        pickle.dump(obj=rfc,file=open(config.model_ojb,'wb'))
        logger.info(f'model Successfull dumped.')

        return rfc , X_test , y_test

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
        self.initiate_model_training()
