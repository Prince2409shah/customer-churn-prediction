import streamlit as st
import pandas as pd 
import joblib 
def to_str_transformer(x):
    return x.astype(str)

churn_model=joblib.load('churn_model.pkl')
decision_threshold_model=joblib.load('decision_threshold.pkl')
st.title('Churn Prediction')
st.write('This application predicts whether a customer is likely to churn based on their service and account information.')
st.sidebar.title('Customer Input')
tenure_months = st.sidebar.number_input(
    "Tenure Months",
    min_value=0,
    step=1
)
monthly_charges=st.sidebar.number_input("Monthly CHarges")
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
# Load training column structure (saved earlier OR recreate once)
all_columns = churn_model.named_steps['preprocessor'].feature_names_in_

# Create empty row with all columns
full_input = pd.DataFrame(columns=all_columns)
full_input.loc[0] = 0

full_input.loc[0, 'Tenure Months'] = tenure_months
full_input.loc[0, 'Monthly Charges'] = monthly_charges
full_input.loc[0, 'Contract'] = contract_type
full_input.loc[0, 'Internet Service'] = internet_service
full_input.loc[0, 'Payment Method'] = payment_method

if st.button("Predict Churn",key="predict_churn_btn"):
    churn_prob = churn_model.predict_proba(full_input)[0][1]
    st.write(f"Churn Probability: {churn_prob:.2f}")
    threshold = decision_threshold_model  # assuming this is a float

    if churn_prob >= threshold:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")
