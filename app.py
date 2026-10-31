
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model.joblib"
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file is missing. Please add model.joblib to the project folder.")
    st.stop()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to estimate the chance of survival.")

st.subheader("Passenger Information")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Gender", ["Female", "Male"], horizontal=True)
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Ticket Fare", 0.0, 500.0, 30.0)

if st.button("Predict Survival", type="primary"):
    try:
        sex_encoded = 1 if sex == "Female" else 0

        input_data = pd.DataFrame(
            [[pclass, sex_encoded, age, fare]],
            columns=FEATURES
        )

        probabilities = model.predict_proba(input_data)[0]
        survival_probability = probabilities[list(model.classes_).index(1)]

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        st.metric(
            "Estimated Chance of Survival",
            f"{survival_probability:.1%}"
        )

        if prediction == 1:
            st.success("The model predicts that the passenger would have survived.")
        else:
            st.warning("The model predicts that the passenger would not have survived.")

    except Exception as error:
        st.error(f"Unable to generate prediction: {error}")

st.caption(
    "Educational example trained on the historical Titanic passenger dataset. "
    "Predictions are estimates and should not be treated as certainty."
)
