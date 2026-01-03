import streamlit as st
import pandas as pd 
import joblib
import numpy as np 
def to_str_transformer(x):
    return x.astype(str)

churn_model=joblib.load('churn_model.pkl')
decision_threshold_model=joblib.load('decision_threshold.pkl')
st.write("MODEL HASH:", hash(str(churn_model)))
st.write("FEATURES:", churn_model.named_steps["preprocessor"].feature_names_in_)
st.title('Churn Prediction')
st.write('This application predicts whether a customer is likely to churn based on their service and account information.')
st.sidebar.title('Customer Input')
tenure_months = st.sidebar.number_input(
    "Tenure Months",
    min_value=0,
    step=1
)
monthly_charges=st.sidebar.number_input("Monthly Charges")
contract_type=st.sidebar.selectbox('Contract type',['Month-to-month','Two year','One year'])
internet_service=st.sidebar.selectbox('Internet Service',['DSL','Fiber optic','No'])
payment_method=st.sidebar.selectbox('Payment Method',['Mailed check','Electronic check','Credit card (automatic)','Bank transfer (automatic)'])
input_data = {
    'Tenure Months': tenure_months,
    'Monthly Charges': monthly_charges,
    'Contract': contract_type,
    'Internet Service': internet_service,
    'Payment Method': payment_method
}
input_df = pd.DataFrame([input_data])
st.write("Input Data Preview")
st.dataframe(input_df)

input_df["Tenure Months"] = input_df["Tenure Months"].astype(float)
input_df["Monthly Charges"] = input_df["Monthly Charges"].astype(float)

for col in ["Contract", "Internet Service", "Payment Method"]:
    input_df[col] = input_df[col].astype(str)

if st.button("Predict Churn",key="predict_churn_btn"):
    churn_prob = churn_model.predict_proba(input_df)[0][1]
    st.write(f"Churn Probability: {churn_prob:.2f}")
    threshold = decision_threshold_model  # assuming this is a float

    if churn_prob >= threshold:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")
