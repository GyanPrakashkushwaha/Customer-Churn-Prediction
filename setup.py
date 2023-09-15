from setuptools import setup , find_packages

with open(file='README.md',mode='r',encoding='utf-8') as f:
    long_des = f.read()

__version__ = '0.0.1'

REPO_NAME = 'Customer-Churn-Prediction'
AUTHOR_USER_NAME = 'GyanPrakashkushwaha'
SRC_REPO = 'churnPredictor'
AUTHOR_EMAIL = 'gyanp7880@gmail.com'


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='Predicts churn of the customers based on few criterias',
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages=find_packages(where=SRC_REPO),
    package_dir={'': SRC_REPO},
    long_description=long_des,
)

