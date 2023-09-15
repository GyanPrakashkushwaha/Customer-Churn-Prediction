from entity import DataTransformationConfig
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder,
                                   MinMaxScaler)
import numpy as np
import joblib
from churnPredictor import logger

class TransformData:
    def __init__(self,config:DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self):
        train_df = pd.read_csv(self.config.train_data)
         
        test_df = pd.read_csv(self.config.test_data)

        train_df['Gender']=train_df['Gender'].replace({'Male':0,'Female':1})
        test_df['Gender']=test_df['Gender'].replace({'Male':0,'Female':1})

        preprocessing = ColumnTransformer(transformers=[
                        ('OHE',OneHotEncoder(drop='first',sparse_output=False,dtype=np.int64),['Location']),
                        ('scaling',MinMaxScaler(),['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB'])
                    ],remainder='passthrough')
        
        transformed_train = preprocessing.fit_transform(train_df)
        transformed_test = preprocessing.fit_transform(test_df)

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
                                        'remainder__Gender':'Gender'})
        
        transformed_test_df = transformed_test_df.rename(columns={
                                        'OHE__Location_Houston': 'Houston',
                                        'OHE__Location_Los Angeles': 'LosAngeles',
                                        'OHE__Location_Miami': 'Miami',
                                        'OHE__Location_New York': 'NewYork',
                                        'scaling__Age': 'Age',
                                        'scaling__Subscription_Length_Months': 'Subscription_Length_Months',
                                        'scaling__Monthly_Bill': 'Monthly_Bill',
                                        'scaling__Total_Usage_GB':'Total_Usage_GB',
                                        'remainder__Gender':'Gender'})
        
        
        transformed_train_df.to_csv(self.config.transform_train_df_path,index=False)
        transformed_test_df.to_csv(self.config.transform_test_df_path,index=False)
        joblib.dump(preprocessing,self.config.preprocessor_obj)
        logger.info("data transformation done!")
        logger.info(f'Columns : {transformed_train_df.columns}')
        logger.info(f'Columns : {transformed_test_df.columns}')
        logger.info(transformed_train_df.shape)
        logger.info(transformed_train_df.shape)
