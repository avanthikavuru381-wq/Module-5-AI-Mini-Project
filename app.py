# AI-Powered Student Performance Predictor
# Module 5 - AI Tools & Mini Project

import os
import pandas as pd
import streamlit as st

from model import train_model


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🎓 AI-Powered Student Performance Predictor")

st.write(
    "Enter student information below to predict the expected final score "
    "using a Machine Learning model."
)


# -----------------------------
# Load Dataset
# -----------------------------

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "student_data.csv"
)

data = pd.read_csv(DATA_PATH)


# -----------------------------
# Train Model
# -----------------------------

model, mae = train_model(data)


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Student Information")

study_hours = st.sidebar.slider(
    "Study Hours per Day",
    min_value=0.0,
    max_value=12.0,
    value=5.0,
    step=0.5
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=80
)

previous_score = st.sidebar.slider(
    "Previous Score",
    min_value=0,
    max_value=100,
    value=70
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours per Day",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.5
)


# -----------------------------
# Prediction
# -----------------------------

input_data = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "sleep_hours": [sleep_hours]
})


prediction = model.predict(input_data)[0]


# -----------------------------
# Display Prediction
# -----------------------------

st.subheader("📊 Prediction Result")

st.metric(
    label="Predicted Final Score",
    value=f"{prediction:.2f}/100"
)


# -----------------------------
# Performance Interpretation
# -----------------------------

if prediction >= 90:
    performance = "Excellent"
elif prediction >= 75:
    performance = "Good"
elif prediction >= 60:
    performance = "Average"
elif prediction >= 40:
    performance = "Needs Improvement"
else:
    performance = "Poor"

st.success(f"Predicted Performance Level: **{performance}**")


# -----------------------------
# Model Information
# -----------------------------

st.subheader("🤖 Model Information")

st.write(
    "This application uses a Random Forest Regression model "
    "trained on student performance data."
)

st.write(f"Model Mean Absolute Error: **{mae:.2f}**")


# -----------------------------
# Input Summary
# -----------------------------

st.subheader("📝 Student Input")

st.dataframe(
    input_data,
    use_container_width=True
)


# -----------------------------
# Footer
# -----------------------------

st.info(
    "This project was developed as part of Module 5 - AI Tools & Mini Project "
    "for the Codomax Digital Solutions internship."
)