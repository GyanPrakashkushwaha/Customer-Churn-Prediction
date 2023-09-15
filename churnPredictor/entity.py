from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir:Path
    data_dir:Path
    schema_check:dict
    make_data:dict