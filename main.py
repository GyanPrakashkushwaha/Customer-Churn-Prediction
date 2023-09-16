
from pipeline.stage_01_data_validation import DataValidationTrainingPipeline
from pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from churnPredictor import CustomException, logger
from pipeline.stage_03_model_trainer import ModelTrainerPipeline 
from pipeline.stage_04_mflow_tracking import MlflowModelTracking

# STAGE_NAME = "Data Validation stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataValidationTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise CustomException(e)


# STAGE_NAME = 'Data Transformation Stage'

# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_transformation = DataTransformationTrainingPipeline()
#    data_transformation.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#    logger.exception(e)
#    raise CustomException(e)



# STAGE_NAME = "Model Training stage"

# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    model_trainer = ModelTrainerPipeline()
#    model_trainer.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#    logger.exception(e)
#    raise CustomException(e)



STAGE_NAME = "Model Tracking Stage stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   mlflow_tracking = MlflowModelTracking()
   mlflow_tracking.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise CustomException(e)
