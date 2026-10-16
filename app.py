import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter the details below to estimate the insurance charges.")

# ---------- User inputs ----------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ---------- Convert inputs to the 11 model columns ----------
is_female = 1 if sex == "Female" else 0
is_smoker = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

bmi_category_healthy = 1 if 18.5 <= bmi < 25 else 0
bmi_category_overweight = 1 if 25 <= bmi < 30 else 0
bmi_category_obese = 1 if bmi >= 30 else 0

if st.button("Predict"):
    data = pd.DataFrame([{
        "age": age,
        "is_female": is_female,
        "bmi": bmi,
        "children": children,
        "is_smoker": is_smoker,
        "region_northwest": region_northwest,
        "region_southeast": region_southeast,
        "region_southwest": region_southwest,
        "bmi_category_healthy": bmi_category_healthy,
        "bmi_category_overweight": bmi_category_overweight,
        "bmi_category_obese": bmi_category_obese,
    }])

    # Make sure columns are in the exact order the model expects
    data = data[list(model.feature_names_in_)]

    result = model.predict(data)
    st.success(f"Estimated insurance charges: ${result[0]:,.2f}")
