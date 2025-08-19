import streamlit as st
import joblib 
import numpy as np

st.title("Salary Prediction Web")

st.divider()


st.write("With this app, you can predict the salary of a person based on their experience and education level.")


years = st.number_input("Enter your years of experience", value=1, step=1, min_value = 0)
jobrate = st.number_input("Enter your job rate",value=3.5,step=0.5, min_value = 0.0, max_value = 5.0)

X = [years, jobrate]

model = joblib.load("salary_prediction_model.pkl")


st.divider()

predict = st.button("Predict")



st.divider()



if predict:

    st.balloons()

    X1 = np.array([X])

    prediction = model.predict(X1)

    st.write(f"The predicted salary is: {prediction[0]}")


else: 
    "Please press the button for app to make a prediction"
