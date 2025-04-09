import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.preprocessing import split_data
from utils.evaluator import evaluate_model
from utils.visualizer import plot_actual_vs_predicted

# You can switch this to allow model selection from the app later
from models.linear_regression_model import train_model

st.title("Real Estate Price Prediction")

st.markdown("Upload a CSV file or use the default dataset.")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data("data/final.csv")  # Use your default CSV
    st.info("Using default dataset.")

if df is not None:
    st.write("### Sample of the data:")
    st.dataframe(df.head())

    if "price" not in df.columns:
        st.error("The dataset must have a 'price' column as the target variable.")
    else:
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test = split_data(df, target_column="price")
                model = train_model(X_train, y_train)
                mse, r2 = evaluate_model(model, X_test, y_test)
                st.success(f" Model trained!")
                st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")
                st.write(f"**R² Score**: {r2:.2f}")

                predictions = model.predict(X_test)
                st.pyplot(plot_actual_vs_predicted(y_test, predictions))
