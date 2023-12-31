from entity import DataTransformationConfig
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder,
                                   MinMaxScaler)
import numpy as np
import joblib
from churnPredictor import logger , CustomException
import pickle


class TransformData:
    def __init__(self,config:DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self):
        train_df = pd.read_csv(self.config.train_data)
        # train_df = train_df.drop(columns='Churn')
         
        test_df = pd.read_csv(self.config.test_data)
        # test_df = test_df.drop(columns='Churn')

        X_train = train_df.drop(columns='Churn')
        y_train = train_df['Churn']
        X_test = test_df.drop(columns='Churn')
        y_test= test_df['Churn']


        X_train['Gender']=X_train['Gender'].replace({'Male':0,'Female':1})
        X_test['Gender']=X_test['Gender'].replace({'Male':0,'Female':1})

        preprocessing = ColumnTransformer(transformers=[
                        ('OHE',OneHotEncoder(drop='first',sparse_output=False,dtype=np.int64),['Location']),
                        ('scaling',MinMaxScaler(),['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB'])
                    ],remainder='passthrough')
        
        transformed_train = preprocessing.fit_transform(X_train)
        transformed_test = preprocessing.transform(X_test)

        transformed_train_df = pd.DataFrame(data=transformed_train,columns=preprocessing.get_feature_names_out())
        transformed_test_df = pd.DataFrame(data=transformed_test,columns=preprocessing.get_feature_names_out())

        transformed_train_df = transformed_train_df.rename(columns={
                                        'OHE__Location_Houston': 'Houston',
                                        'OHE__Location_Los Angeles': 'LosAngeles',
                                        'OHE__Location_Miami': 'Miami',
                                        'OHE__Location_New York': 'NewYork',
                                        'scaling__Age': 'Age',
                                        'scaling__Subscription_Length_Months': 'Subscription_Length_Months',
                                        'scaling__Monthly_Bill': 'Monthly_Bill',
                                        'scaling__Total_Usage_GB':'Total_Usage_GB',
                                        'remainder__Gender':'Gender',
                                        'remainder__Churn': 'Churn'})
        
        transformed_test_df = transformed_test_df.rename(columns={
                                        'OHE__Location_Houston': 'Houston',
                                        'OHE__Location_Los Angeles': 'LosAngeles',
                                        'OHE__Location_Miami': 'Miami',
                                        'OHE__Location_New York': 'NewYork',
                                        'scaling__Age': 'Age',
                                        'scaling__Subscription_Length_Months': 'Subscription_Length_Months',
                                        'scaling__Monthly_Bill': 'Monthly_Bill',
                                        'scaling__Total_Usage_GB':'Total_Usage_GB',
                                        'remainder__Gender':'Gender',
                                        'remainder__Churn': 'Churn'})
        
        
        transformed_train_df.to_csv(self.config.transform_X_train_path,index=False)
        transformed_test_df.to_csv(self.config.transform_X_test_path,index=False)
        y_train.to_csv(self.config.y_train_path,index=False)
        y_test.to_csv(self.config.y_test_path,index=False)
        
        joblib.dump(preprocessing,self.config.preprocessor_obj)
        logger.info("data transformation done!")
        logger.info(f'Columns : {transformed_train_df.columns}')
        logger.info(f'Columns : {transformed_test_df.columns}')
        logger.info(transformed_train_df.shape)
        logger.info(transformed_train_df.shape)
