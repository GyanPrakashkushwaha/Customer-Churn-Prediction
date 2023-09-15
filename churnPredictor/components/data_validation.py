from entity import DataValidationConfig
from churnPredictor import CustomException,logger
import pandas as pd

class DataValidataion:
    def __init__(self,config_in :DataValidationConfig):
        self.config_in = config_in
    
    def validate_features(self) -> bool:
        try:
            validation_status = False
            df = pd.read_csv(self.config_in.data_dir)
            df = df.drop(columns='Churn')
            all_features = list(df.columns)
            all_schema = self.config_in.schema_check

            for cols in all_features:
                if cols not in all_schema:
                    validation_status = False
                    with open(self.config_in.STATUS_FILE,'w') as stat_file:
                        stat_file.write(f"Validation status: {validation_status}")
                
                else:
                    validation_status = True
                    with open(self.config_in.STATUS_FILE,'w') as f:
                        f.write(f"Validation status: {validation_status}")

            logger.info(f'Validation Status is {validation_status}')
            return validation_status
        except Exception as e:
            raise CustomException(e)
        