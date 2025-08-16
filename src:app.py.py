import streamlit as st
import joblib
import pandas as pd
import shap


diabetes_model = joblib.load('models/diabetes_model.pkl')
heart_model = joblib.load('models/heart_model.pkl')

diabetes_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigree', 'Age']
heart_features = ['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'Fbs', 
                 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal']

st.title('Healthcare Disease Prediction System')

disease = st.sidebar.selectbox('Select Disease', ['Diabetes', 'Heart Disease'])

if disease == 'Diabetes':
    st.header('Diabetes Risk Prediction')
    input_data = []
    for feature in diabetes_features:
        input_data.append(st.number_input(feature, min_value=0.0, step=0.1))
    
    if st.button('Predict Diabetes Risk'):
        df = pd.DataFrame([input_data], columns=diabetes_features)
        prediction = diabetes_model.predict(df)[0]
        probability = diabetes_model.predict_proba(df)[0][1]
        
        st.success(f'Risk Probability: {probability:.2%}')
        st.subheader('Explanation:')
        
        
        explainer = shap.TreeExplainer(diabetes_model)
        shap_values = explainer.shap_values(df)
        st.pyplot(shap.force_plot(explainer.expected_value, shap_values[0], df))

