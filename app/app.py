import streamlit as st
import pandas as pd
import joblib
import os

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the trained model and scaler
model = joblib.load(
    os.path.join(BASE_DIR, "models", "fetal_health_gradient_boosting.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "fetal_health_scaler.pkl")
)

print("Model and scaler loaded successfully!")

# App title
st.set_page_config(
    page_title="Fetal Health Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Multiclass Classification of Fetal Health")
st.write(
    "Predict fetal health status from Cardiotocogram (CTG) measurements "
    "using a Gradient Boosting classifier."
)

st.info(
    "Academic demonstration only. This system is not intended for clinical diagnosis."
)

# Load cleaned dataset to get realistic input ranges and default values
data_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "fetal_health_clean.csv"
)

clean_data = pd.read_csv(data_path)

features = [
    "LB", "AC.1", "FM.1", "UC.1",
    "DL.1", "DS.1", "DP.1",
    "ASTV", "MSTV", "ALTV", "MLTV",
    "Width", "Min", "Max", "Nmax", "Nzeros",
    "Mode", "Mean", "Median", "Variance", "Tendency"
]

st.subheader("Enter CTG Measurements")

st.write("Enter the 21 Cardiotocogram (CTG) feature values:")

input_values = {}

# Display inputs in 3 columns
columns = st.columns(3)

for i, feature in enumerate(features):
    with columns[i % 3]:
        input_values[feature] = st.number_input(
            feature,
            min_value=float(clean_data[feature].min()),
            max_value=float(clean_data[feature].max()),
            value=float(clean_data[feature].median()),
            step=0.1
        )

# Prediction button
if st.button("🔍 Predict Fetal Health", type="primary"):

    # Convert user inputs into a DataFrame
    input_data = pd.DataFrame([input_values])

    # Keep features in the exact order used during training
    input_data = input_data[features]

    # Scale the input using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🟢 Normal Fetal Health")
    elif prediction == 2:
        st.warning("🟡 Suspect Fetal Health")
    else:
        st.error("🔴 Pathological Fetal Health")