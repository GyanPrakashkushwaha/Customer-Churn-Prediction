


from components.mlflow_tracking import TrackModelPerformance
from configuration import ConfigurationManager
from churnPredictor import CustomException , logger


class MlflowModelTracking:
    def __init__(self) :
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            mlflow_tracking_config = config.get_mlflow_tracking_config()
            track_model = TrackModelPerformance(config=mlflow_tracking_config)
            track_model.start_mlflow()
        except Exception as e:
            logger.info(f'Exception raised [{e}]')
            raise CustomException(e)
