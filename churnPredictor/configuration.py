from entity import (DataTransformationConfig,
                     DataValidationConfig, 
                     MLFlowTrackingConfig, 
                     ModelTrainerConfig)
from churnPredictor.constants import *
from churnPredictor.utils import *


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):


        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_dirs([self.config.artifacts_root])

    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.INDENPENT_FEATURES

        create_dirs([config.root_dir])
        

        data_validation_config =  DataValidationConfig(
            root_dir=config.root_dir,
            data_dir=config.data_dir,
            schema_check = schema,
            make_data=config.make_data,
            STATUS_FILE=config.status_file
        )

        return data_validation_config
        
        
    
    def get_data_transformation_config(self):
        config = self.config.data_transformation
        # schema = self.schema.columns_renamer

        create_dirs([config.root_dir,config.model_dir])
        

        return DataTransformationConfig(
            root_dir=config.root_dir,
            train_data=config.train_data_path,
            test_data=config.test_data_path,
            transform_test_df_path=config.transformed_test_df_path,
            transform_train_df_path=config.transformed_train_df_path,
            preprocessor_obj=config.preprocessor_obj,
            model=config.model_dir)
        
     
    def get_modelTrainer_config(self):
        config = self.config.model_trainer
        params = self.params.RandomForest

        create_dirs([config.model_dir])

        return ModelTrainerConfig(
            train_data=config.train_data,
            test_data=config.test_data,
            model_dir=config.model_dir,
            model_ojb=config.model_obj,
            n_estimators=params.n_estimators,
            oob_score=params.oob_score)
    

    def get_mlflow_tracking_config(self) -> MLFlowTrackingConfig:
        config = self.config.mlflow_tracking
        params = self.params.RandomForest
        schema = self.schema.DEPENDET_FEATURES

        # create_dirs([self.config.mflow_dir])

        return MLFlowTrackingConfig(
            mflow_dir=config.mlflow_dir,
            test_data=config.test_data,
            model_obj=config.model_obj_path,
            metrics_file=config.metrics_file_name,
            params=params,
            target_col=schema.Churn,
            mlflow_uri='https://dagshub.com/GyanPrakashKushwaha/Customer-Churn-Prediction.mlflow',
            confusion_metrics=config.confusion_metrics
        )
