import streamlit as st
import pandas as pd
import joblib

st.title('Customer churn Predictor')

df = pd.read_csv(r'artifacts\raw_data\customer_churn_removed_col.csv')
model = joblib.load(open(f'artifacts\model\RandomForestClassifier.joblib', 'rb'))
preprocess = joblib.load(open(f'artifacts\model\preprocessorObj.joblib', 'rb'))

age = st.number_input(label='Age', min_value=18, max_value=70)
gender = st.selectbox(label='Gender', options=['male', 'female'])
location = st.selectbox(label='Location', options=['Houston', 'LosAngeles', 'Miami', 'NewYork'])  # Ensure exact match
subscription_length = st.number_input(label='Subscription Length (in Month)', min_value=1, max_value=24)
monthly_bill = st.number_input(label='Monthly_Bill', min_value=30, max_value=100)
total_usage = st.number_input(label='Total Usage in GB', min_value=50, max_value=500)

if gender == 'male':
    gender_trans = 0
else:
    gender_trans = 1

input_data = pd.DataFrame(data=[[age, gender_trans, subscription_length, monthly_bill, total_usage, location]],
                          columns=['Age', 'Gender', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB', 'Location'])


input_data_transformed = preprocess.transform(input_data)
# st.write(input_data_transformed)
# st.write(input_data)
if st.button('Predict'):
    pred = model.predict(input_data_transformed)[0]
    if pred ==0:
        st.subheader(f'Customer went [{pred} Customers churn ]')
    else:
        st.subheader(f'Customer stayed [{pred} Customers churn ]')

