import streamlit as st
import joblib
import pandas as pd

data = pd.read_csv("Dataset/Salary_data.csv")
model = joblib.load("Model/model.pkl")

def predictSalary(num):
    return model.predict([[num]])[0][0].round(3)

if __name__ == '__main__':
    st.sidebar.text("Sample Data")
    st.sidebar.dataframe(data = data)

    st.title("Salary Predictor by Experience")

    years = st.number_input("Enter your total years of work experience", max_value=80.0, step=1.,format="%.2f")
    submit = st.button("Predict", type="primary")

    if(submit and years):
        st.write(f"Your expected salary as per your {years} years of work experience is: {predictSalary(float(years))} (approx)")