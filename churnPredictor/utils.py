from ensure import ensure_annotations
from pathlib import Path
import yaml
from churnPredictor import logger, CustomException
import sys
import os
import json     
from box import ConfigBox
import joblib
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as f:
            content = yaml.safe_load(f)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(error_msg=e, error_detail=sys)
        

@ensure_annotations
def create_dirs(path_to_dirs:list,verbose=True):
    for path in path_to_dirs:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f'Created {path}')


@ensure_annotations
def save_json(path:Path,data:dict):
    with open(path,'r') as f:
        json.dump(data,f,indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path:Path) -> ConfigBox:
    with open(path,'r') as f:
        logger.info(f"json file loaded succesfully from: {path}")
        return ConfigBox(json.load(f))
    


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"



