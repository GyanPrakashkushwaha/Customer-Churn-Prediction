from components.data_transformation import TransformData
from configuration import ConfigurationManager


STAGE_NAME = 'Data Transformation Stage'

class DataTransformationTrainingPipeline:
    def __init__(self) :
        pass

    def main(self):
        config = ConfigurationManager()
        entity = ConfigurationManager()
        get_entity = entity.get_data_transformation_config()
        trans_data = TransformData(config=get_entity)
        trans_data.initiate_data_transformation()