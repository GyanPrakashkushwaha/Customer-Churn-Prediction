## Customer Churn Prediction using Machine Learning and Deep learning.

> #### Done ✅-> EDA✅ , Data Validation✅ , Model Training✅, Model Flow Tracking✅


## Data Insights:
- #### Numerical Data Points are equally distributed.
![EDA Image](artifacts/readme/EDA.png)
- #### Almost equally distribution of each class in each feature.
![Cat Features](artifacts/readme/categorical_features.png)
- #### With respect to churn
equal contribution of all classes
![with repect to churn](artifacts/readme/output.png)


### Installation
1. Clone the repository:
   
```python
git clone https://github.com/GyanPrakashkushwaha/Customer-Churn-Prediction.git customer-churn-prediction
```

2. Navigate to the project directory:

```python
cd customer-churn-prediction
```
3. Create virtaul environment and activate it.
```python
virtalenv churnvenv 
churnvenv/Scipts/activate.ps1
```

3. Install the required dependencies:
```python
pip install -r requirements.txt
```
4. run main.py for `data validation` , `data transformation`, `model training` and `mlflow tracking`.
```Python
python run main.py
```  

5. Run the the streamlit app:
```python
streamlit run app.py
```

### MLflow 

- MLflow for local web server
```Python
mlflow ui
```

- run this in environment 
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/GyanPrakashKushwaha/Customer-Churn-Prediction.mlflow
export MLFLOW_TRACKING_USERNAME=GyanPrakashKushwaha 
export MLFLOW_TRACKING_PASSWORD=53950624aa84e08b2bd1dfb3c0778ff66c4e7d05
```
- Tracking URL
```Python
https://dagshub.com/GyanPrakashKushwaha/Customer-Churn-Prediction.mlflow
```

#### required packages:
```Python
pandas
pyYAML
tqdm
ensure==1.0.2
joblib
python-box==6.0.2
scikit-learn
dagshub
mlflow==2.2.2
seaborn
streamlit
```



## TODO
- read data from mondoDB 
- deploy the model in AWS