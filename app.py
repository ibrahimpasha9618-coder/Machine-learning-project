import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="My ML Project", page_icon="🤖")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("My ML Project")
st.write("Enter the details below to get a prediction.")

# CHANGE these to match your own features
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)

if st.button("Predict"):
    data = pd.DataFrame([[feature1, feature2]],
                        columns=["feature1", "feature2"])
    result = model.predict(data)
    st.success(f"Prediction: {result[0]}")
