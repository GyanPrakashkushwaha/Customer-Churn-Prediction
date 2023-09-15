from entity import DataTransformationConfig, DataValidationConfig
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
        
        