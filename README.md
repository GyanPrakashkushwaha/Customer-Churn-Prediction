# Customer Churn Prediction using Machine Learning and Deep learning.

# EDA 
- ## Numerical Data Points are equally distributed.
![EDA Image](artifacts/readme/EDA.png)
- ## Almost equally distribution of each class in each feature.
![Cat Features](artifacts/readme/categorical_features.png)
- ## With respect to churn
equal contribution of all classes
![with repect to churn](artifacts/readme/output.png)








## MLflow command for local web server
```Python
mlflow ui
```

## run this in environment 
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/GyanPrakashKushwaha/Customer-Churn-Prediction.mlflow
export MLFLOW_TRACKING_USERNAME=GyanPrakashKushwaha 
export MLFLOW_TRACKING_PASSWORD=53950624aa84e08b2bd1dfb3c0778ff66c4e7d05
```


## TODO: 
- read data from mongoDB
- do EDA
- add joblib model
- add max_value