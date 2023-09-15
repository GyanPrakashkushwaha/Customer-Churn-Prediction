from components.model_trainer import ModelTrainer
from configuration import ConfigurationManager


STAGE_NAME = "Model Training stage"


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        trainer_config = config.get_modelTrainer_config()
        model_trainer = ModelTrainer(config=trainer_config)
        model_trainer.train_model()




# try:
#     config = ConfigurationManager()
#     trainer_config = config.get_modelTrainer_config()
#     model_trainer = ModelTrainer(config=trainer_config)
#     model_trainer.train_model()
# except Exception as e:
#     raise CustomException(e)
