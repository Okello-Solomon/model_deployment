import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# Load the trained model
model = joblib.load('XGBClassifier.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Set up the Streamlit app
st.title('Loan Payback Prediction')
st.subheader('A Machine Learning-Based Loan Repayment Classifier')
st.write(
    'Enter the applicant’s loan and financial details to predict whether the loan will be paid back.'
)


# Create input fields for user to enter loan applicant details

gender = st.selectbox('Gender', ['Male', 'Female', 'Others'])

marital_status = st.selectbox(
    'Marital Status',
    ['Single', 'Married', 'Divorced', 'Widowed']
)

education_level = st.selectbox(
    'Education Level',
    ['High School', "Bachelor's", "Master's", 'PhD', 'Other']
)

employment_status = st.selectbox(
    'Employment Status',
    ['Employed', 'Self_Employed', 'Unemployed', 'Stendent', 'Retired']
)

loan_purpose = st.selectbox(
    'Loan Purpose',
    ['Business', 'Car', 'Debt_consolidation','Education', 'Home', 'Medical', 'Personal', 'Other', 'Vacation']
)

grade_subgrade = st.selectbox(
    'Loan Grade (Risk Level)',
    ['A1','A2','A3','A4','A5',
     'B1','B2','B3','B4','B5',
     'C1','C2','C3','C4','C5',
     'D1','D2','D3','D4','D5',
     'E1','E2','E3','E4','E5',
     'F1','F2','F3','F4','F5']
)

annual_income = st.number_input(
    'Annual Income',
    min_value=0.00,
    max_value=1_000_000.00,
    value=50_000.00,
    format="%.2f" 
)

loan_amount = st.number_input(
    'Loan Amount',
    min_value=0.0,
    max_value=500_000.0,
    value=10_000.0,
    format="%.2f" 
)

interest_rate = st.number_input(
    'Interest Rate (%)',
    min_value=0.0,
    max_value=40.0,
    value=12.5,
    format="%.2f" 
)

debt_to_income_ratio = st.number_input(
    'Debt-to-Income Ratio',
    min_value=0.00,
    max_value=1.00,
    value=0.30,
    format="%.3f" 
)

credit_score = st.slider(
    'Credit Score',
    min_value=300,
    max_value=850,
    value=650
)

# Create button for prediction
if st.button('Predict Loan Payback'):

    # Dummy encode the embarked variable
    #Gender
    gender_Male = 0
    gender_Other = 0
    if gender == 'Male':
        gender_Male = 1
    elif gender == 'Others':
        gender_Other = 1
    # gender_Female is the reference category so we don't need to create dummy variable for it Female-> Other=0, Male=0

    # Marital Status
    marital_status_Married = 0
    marital_status_Single = 0
    marital_status_Widowed = 0
    if marital_status == 'Married':
        marital_status_Married = 1
    elif marital_status == 'Single':
        marital_status_Single = 1
    elif marital_status == 'Widowed':
        marital_status_Widowed = 1
    # marital_status_Divorced is the reference category so Divorced -> Married=0,Single=0,Widowed=0 
    
    # Employment Status
    employment_status_Retired = 0
    employment_status_Self_employed = 0
    employment_status_Student = 0
    employment_status_Unemployed = 0

    if employment_status == 'Retired':
        employment_status_Retired = 1
    elif employment_status == 'Self_employed':
        employment_status_Self_employed = 1
    elif employment_status == 'Student':
        employment_status_Student = 1
    elif employment_status == 'Unemployed':
        employment_status_Unemployed = 1
    # Employed (reference) -> all 0


    # Loan Purpose
    # Initialize all loan purpose dummies
    loan_purpose_Car = 0
    loan_purpose_Debt_consolidation = 0
    loan_purpose_Education = 0
    loan_purpose_Home = 0
    loan_purpose_Medical = 0
    loan_purpose_Other = 0
    loan_purpose_Vacation = 0

    # Set user input
    if loan_purpose == 'Car':
        loan_purpose_Car = 1
    elif loan_purpose == 'Debt Consolidation': 
        loan_purpose_Debt_consolidation = 1
    elif loan_purpose == 'Education':
        loan_purpose_Education = 1
    elif loan_purpose == 'Home':
        loan_purpose_Home = 1
    elif loan_purpose == 'Medical':
        loan_purpose_Medical = 1
    elif loan_purpose == 'Other':
        loan_purpose_Other = 1
    elif loan_purpose == 'Vacation':
        loan_purpose_Vacation = 1

 

    # Ordinal mappings for Streamlit inputs 
    education_level_map = {
        'High School': 1,
        "Bachelor's": 2,
        "Master's": 3,
        'PhD': 4,
        'Other': 5
    }

    grade_subgrade_map = {
        'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5,
        'B1': 6, 'B2': 7, 'B3': 8, 'B4': 9, 'B5': 10,
        'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15,
        'D1': 16, 'D2': 17, 'D3': 18, 'D4': 19, 'D5': 20,
        'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25,
        'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 30
    }

    # Assign numeric values from user input 
    education_level = education_level_map[education_level]
    grade_subgrade = grade_subgrade_map[grade_subgrade]


    # Initialize all dummy variables to 0 
    input_data = pd.DataFrame([[
    annual_income,
    debt_to_income_ratio,
    credit_score,
    loan_amount,
    interest_rate,
    education_level,
    grade_subgrade,
    gender_Male,
    gender_Other,
    marital_status_Married,
    marital_status_Single,
    marital_status_Widowed,
    employment_status_Retired,
    employment_status_Self_employed,
    employment_status_Student,
    employment_status_Unemployed,
    loan_purpose_Car,
    loan_purpose_Debt_consolidation,
    loan_purpose_Education,
    loan_purpose_Home,
    loan_purpose_Medical,
    loan_purpose_Other,
    loan_purpose_Vacation

]], columns = feature_names) 

    #Scale the input data
    input_data_scaled = scaler.transform(input_data)
    # Make the prediction
    prediction = model.predict(input_data_scaled)

    #Display the prediction result
    if prediction[0] == 1:
        st.success('The loan is likely to be paid back.')
    else:
        st.error('High risk: The loan may not be repaid.')
