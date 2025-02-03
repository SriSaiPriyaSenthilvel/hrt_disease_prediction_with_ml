import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Data
data = pd.read_csv("heart(1).csv")

# Prepare Data
X = data.drop(columns='target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Streamlit UI Styling
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.markdown(
    """
    <style>
    
    body {
        background-color: #e6f7e6;
        color: #2d6a4f;
        font-family: 'comic sans';
    }
    h1, h2 {
        font-weight: bold;
        color: #1b4332;
    }
    .stButton>button {
        background-color: #40916c;
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 12px;
        border-radius: 12px;
    }
    .stTextInput input, .stNumberInput input {
        font-weight: bold;
    }
    .stRadio input {
        font-weight: bold;
    }
    .stSlider div {
        font-weight: bold;
    }
    .stSubheader {
        font-weight: bold;
    }
    .stSuccess, .stError {
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create Columns Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.title("Heart Disease Prediction App")
    st.subheader("Enter your health details below:")

    # User Inputs
    age = st.number_input("Age", min_value=20, max_value=100, value=40)
    sex = st.radio("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    fbs = 1 if fbs == "Yes" else 0
    restecg = st.slider("Resting ECG Results (0-2)", 0, 2, 1)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
    exang = 1 if exang == "Yes" else 0
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.slider("Slope of Peak Exercise ST Segment (0-2)", 0, 2, 1)
    ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
    thall = st.slider("Thalassemia (0-3) [Higher is Riskier]", 0, 3, 1)

    # Correct Misinterpretation by Inverting Thalassemia Score
    thall = 3 - thall

    # Prediction
    if st.button("🔍 Predict"):
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thall])
        reshaped_data = input_data.reshape(1, -1)
        scaled_data = scaler.transform(reshaped_data)
        prediction_prob = model.predict_proba(scaled_data)[0][1] * 100

        st.subheader("Prediction Result:")
        if prediction_prob >= 50:
            st.error(f"⚠️ High Risk of Heart Disease! (Risk: {prediction_prob:.2f}%)")
            risk_levels = ["High Risk"]
            risk_values = [prediction_prob]
            colors = ['red']
        else:
            st.success(f"✅ Low Risk, Healthy Heart! (Risk: {prediction_prob:.2f}%)")
            risk_levels = ["Low Risk"]
            risk_values = [100 - prediction_prob]
            colors = ['green']

        # Visualization with Bar Chart
        st.subheader("Risk Probability Visualization:")
        fig, ax = plt.subplots()
        ax.bar(risk_levels, risk_values, color=colors)
        ax.set_ylabel("Probability (%)")
        ax.set_ylim([0, 100])
        ax.set_title("Heart Disease Risk Assessment")
        st.pyplot(fig)

        # Medication and Lifestyle Recommendations
        st.subheader("Recommendations to Manage Risk:")
        if prediction_prob >= 50:
            st.write("- Follow a heart-healthy diet rich in fruits, vegetables, and whole grains.")
            st.write("- Engage in at least 30 minutes of moderate exercise daily.")
            st.write("- Maintain a healthy weight and avoid smoking.")
            st.write("- Manage stress through meditation or yoga.")
            st.write("- Regularly monitor blood pressure and cholesterol levels.")
            st.write("- Consult a doctor for potential medication if necessary.")
        else:
            st.write("- Continue a balanced diet and regular physical activity.")
            st.write("- Keep monitoring health parameters to stay in good condition.")

with col2:
    st.image("hrt.jpg", use_container_width=True)
