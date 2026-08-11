import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Load Model and Preprocessing
# -----------------------------
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's details to predict the heart disease result.")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Predict", "About"]
)

# -----------------------------
# Home Page
# -----------------------------
if page == "Home":

    st.header("Welcome 👋")

    st.write(
        "This Machine Learning application predicts the possibility "
        "of heart disease using patient medical information."
    )

    st.info(
        "Go to the 'Predict' section from the sidebar and enter "
        "the required patient details."
    )

# -----------------------------
# Prediction Page
# -----------------------------
elif page == "Predict":

    st.header("🩺 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50,
            step=1
        )

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=80,
            max_value=200,
            value=120,
            step=1
        )

        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=200,
            step=1
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1]
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=60,
            max_value=220,
            value=150,
            step=1
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1]
        )

        oldpeak = st.number_input(
            "ST Depression",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope of ST Segment",
            [0, 1, 2]
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thalassemia",
            [1, 2, 3]
        )

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🔍 Predict", width="stretch"):

        input_data = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        try:

            # Apply scaler
            input_scaled = scaler.transform(input_data)

            # Prediction
            prediction = model.predict(input_scaled)

            st.subheader("Prediction Result")

            if prediction[0] == 1:
                st.error(
                    "⚠️ The model predicts a higher likelihood of heart disease."
                )
            else:
                st.success(
                    "✅ The model predicts a lower likelihood of heart disease."
                )

        except Exception as e:

            st.error("Prediction could not be completed.")
            st.exception(e)

# -----------------------------
# About Page
# -----------------------------
elif page == "About":

    st.header("📌 About This App")

    st.markdown("""
    ### Heart Disease Prediction

    This application uses Machine Learning to predict the possibility
    of heart disease based on patient information.

    **Technologies Used:**
    - Python
    - Streamlit
    - NumPy
    - Scikit-learn
    - Joblib

    **Features Used:**
    1. Age
    2. Sex
    3. Chest Pain Type
    4. Resting Blood Pressure
    5. Cholesterol
    6. Fasting Blood Sugar
    7. Resting ECG
    8. Maximum Heart Rate
    9. Exercise Induced Angina
    10. ST Depression
    11. Slope
    12. Number of Major Vessels
    13. Thalassemia
    """)


