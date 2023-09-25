from components.mlflow_tracking import TrackModelPerformance
from configuration import ConfigurationManager
from churnPredictor import CustomException , logger
from components.mlflow_serving import ModelServing

class MLFlowModelServing:
    def __init__(self) :
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            mlflow_tracking_config = config.get_mlflow_tracking_config()
            serve_model = ModelServing(config=mlflow_tracking_config)
            serve_model.start_mlflow()
        except Exception as e:
            logger.info(f'Exception raised [{e}]')
            raise CustomException(e)
