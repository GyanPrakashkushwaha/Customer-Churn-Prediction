from components.data_validation import DataValidataion
from configuration import ConfigurationManager


STAGE_NAME = "Data Validation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_val_config = config.get_data_validation_config()
        data_validation = DataValidataion(config_in=data_val_config)
        data_validation.validate_features()

