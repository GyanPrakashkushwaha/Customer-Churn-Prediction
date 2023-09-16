import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


list_of_files = [
    ".github/workflows/.gitkeep",
    f"churnPredictor/__init__.py",
    f"churnPredictor/components/__init__.py",
    f"churnPredictor/utils.py",
    f"churnPredictor/config.py",
    f"churnPredictor/configuration.py",
    f"churnPredictor/pipeline/__init__.py",
    f"churnPredictor/entity.py",
    f"churnPredictor/constants.py",
    "config/config.yaml",
    "main.py",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "experiments/trials.ipynb",
    "templates/index.html",
    'artifacts/data',
    'schema.yaml',
    'app.py'
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")
