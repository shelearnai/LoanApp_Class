import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
def load_model():
    with open('loan_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to accept user data
def user_input_features():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income', min_value=0, value=0)
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0, value=0)
    #loan_amount = st.number_input('Loan Amount', min_value=0, value=0)
    loan_amount_term = st.number_input('Loan Amount Term', min_value=0, value=360)
    credit_history = st.selectbox('Credit History', ['0', '1'])
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
    loan_status = st.selectbox('Loan Status', ['Yes', 'No'])

    data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        #'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
        'Loan_Status': loan_status
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Function to transform categorical features to numeric
def transform_features(features):
    
    features['Gender'] = features['Gender'].map({'Male': 1, 'Female': 0})
    features['Married'] = features['Married'].map({'Yes': 1, 'No': 0})
    features['Dependents'] = features['Dependents'].replace('3+', 3).astype(float)
    features['Education'] = features['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    features['Self_Employed'] = features['Self_Employed'].map({'Yes': 1, 'No': 0})
    features['Credit_History'] = features['Credit_History'].astype(float)
    features['Property_Area'] = features['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    features['Loan_Status'] = features['Loan_Status'].map({'Yes': 1, 'No': 0})

    return features
    
def main():
    st.title('Loan Prediction Web App')
    st.write("Please enter the following information to predict loan status")

    model=load_model()
    features=user_input_features()
    features=transform_features(features)

    if st.button('Predict'):
        
        
        pred_value = model.predict(features)
        st.subheader('Prediction Result')
        st.write(f"The model predicts the loan amount will be {np.round(pred_value[0],2)}")

if __name__ == '__main__':
    main()




