
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
model_filename = '/content/drive/MyDrive/Diabetes_Prediction/rf_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

def main():
    st.sidebar.header("Diabetes Risk Prediction")

    # Input number for user data
    Pregnancies = st.number_input("Input Your Number of Pregnancies", min_value=0, max_value=16, step=1)
    Glucose = st.number_input("Input your Glucose", min_value=74, max_value=200, step=1)
    BloodPressure = st.number_input("Input your Blood Pressure", min_value=30, max_value=130, step=1)
    SkinThickness = st.number_input("Input your Skin Thickness", min_value=0, max_value=100, step=1)
    Insulin = st.number_input("Input your Insulin", min_value=0, max_value=400, step=1)
    BMI = st.number_input("Input your BMI", min_value=14.0, max_value=60.0, step=0.1, format="%.1f")
    DPF = st.number_input("Input your Diabetes Pedigree Function", min_value=0.000, max_value=6.000, step=0.001, format="%.3f")
    Age = st.number_input("Input your Age", min_value=0, max_value=100, step=1)

    # Prepare the inputs for the model
    inputs = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DPF': [DPF],
        'Age': [Age]
    })

    if st.button('Predict'):
        # Make prediction
        result = loaded_model.predict(inputs)
        probabilities = loaded_model.predict_proba(inputs)

        if result[0] == 0:
            st.write("Low risk of diabetes")
        else:
            st.write("Higher risk of diabetes")

        st.subheader('Prediction Probabilities')
        st.write(f"Probability of Non-Diabetic: {probabilities[0][0]:.2f}")
        st.write(f"Probability of Diabetic: {probabilities[0][1]:.2f}")

        st.subheader('Data')
        st.write(inputs)

if __name__ == '__main__':
    main()
