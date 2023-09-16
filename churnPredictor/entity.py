from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir:Path
    data_dir:Path
    schema_check:dict
    make_data:dict
    STATUS_FILE:str


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir : Path
    train_data: Path
    test_data: Path
    transform_X_train_path : Path
    y_train_path : Path
    transform_X_test_path : Path
    y_test_path : Path
    preprocessor_obj : str
    model : Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    train_data : Path
    test_data : Path
    model_dir : Path
    y_train_path : Path
    y_test_path : Path
    model_params_dir : dict

@dataclass(frozen=True)
class MLFlowTrackingConfig:
    mflow_dir : Path
    test_data: Path
    model_obj : str
    metrics_file: str
    params : dict
    mlflow_uri : str
    target_col : str
    confusion_metrics:str
    y_test_path: Path

    

    