import streamlit as st
import pandas as pd
import joblib


st.title('Customer churn Predictor')

df = pd.read_csv(r'artifacts\raw_data\customer_churn_removed_col.csv')
model = joblib.load(open(f'artifacts\model\model.joblib','rb'))
preprocess = joblib.load(open(f'artifacts\model\preprocessorObj.joblib','rb'))
# st.write(df)
age = st.number_input(label='Age',min_value=18,max_value=70)
gender = st.selectbox(label='Gender',options=['male','female'])
location = st.selectbox(label='Location',options=df['Location'].unique())
subscirption_lenght = st.number_input(label='Subscription Length (in Month)',min_value=1,max_value=24)
monthly_bill = st.number_input(label='Monthly_Bill',min_value=30,max_value=100)
total_usage = st.number_input(label='Total Usage in GB',min_value=50,max_value=500)

input_data = preprocess.transform([[age,gender,location,subscirption_lenght,monthly_bill,total_usage]])
st.write(input_data)
st.subheader(f'Customers churn: {model.predict(input_data)}')
st.write(model)