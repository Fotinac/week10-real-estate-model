
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="🏘️ Real Estate Price Predictor", layout="centered")

st.markdown("<h2 style='text-align: center;'>🏘️ Real Estate Price Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict housing prices based on real property features.</p>", unsafe_allow_html=True)
st.divider()

# Load pre-trained model
model = joblib.load("models/real_estate_model.pkl")

st.markdown("### 🧾 Enter Property Details for Prediction")

# Numeric inputs
year_sold = st.number_input("Year Sold", min_value=2000, max_value=2030, value=2024)
property_tax = st.number_input("Property Tax ($)", value=3500.0)
insurance = st.number_input("Insurance ($)", value=1200.0)
beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
baths = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
sqft = st.number_input("Square Footage", value=1800.0)
year_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2005)
lot_size = st.number_input("Lot Size (sqft)", value=4000.0)
property_age = st.number_input("Property Age (Years)", value=19)

# Binary/categorical inputs
basement = st.radio("Basement", ["No", "Yes"])
popular = st.radio("Popular Location", ["No", "Yes"])
recession = st.radio("Sold During Recession", ["No", "Yes"])
property_type = st.selectbox("Property Type", ["Bunglow", "Condo"])

# Encode binary & one-hot
input_data = {
    "year_sold": year_sold,
    "property_tax": property_tax,
    "insurance": insurance,
    "beds": beds,
    "baths": baths,
    "sqft": sqft,
    "year_built": year_built,
    "lot_size": lot_size,
    "basement": 1 if basement == "Yes" else 0,
    "popular": 1 if popular == "Yes" else 0,
    "recession": 1 if recession == "Yes" else 0,
    "property_age": property_age,
    "property_type_Bunglow": 1 if property_type == "Bunglow" else 0,
    "property_type_Condo": 1 if property_type == "Condo" else 0
}

input_df = pd.DataFrame([input_data])

if st.button("🎯 Predict Price"):
    prediction = model.predict(input_df)[0]
    st.markdown("### 💰 Predicted Price")
    st.success(f"${prediction:,.2f}")

    st.markdown("This prediction is based on your input values and a pre-trained linear regression model.")

    # Plot: Visual comparison bar
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Predicted Price"], [prediction], color='steelblue')
    ax.set_ylabel("Price ($)")
    ax.set_title("Prediction Visualization")
    st.pyplot(fig)

