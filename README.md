
## Live App  
[Click to open the app](https://week10-real-estate-model-d34ydl8cjghghgjwgc4d44.streamlit.app) 

---
# Real Estate Price Prediction Application

This app has been built using Streamlit and is designed to predict real estate prices based on various input features.

## Overview
This application predicts real estate prices leveraging machine learning models trained on comprehensive real estate data.

## Features
- User-friendly interface powered by Streamlit.
- Input form to enter details relevant to real estate properties such as location, size, number of rooms, etc.
- Real-time prediction of property prices based on the trained model.

## Dataset
The application utilizes a curated real estate dataset (`final.csv`) that includes features such as:
- Property size
- Location details
- Number of bedrooms and bathrooms
- Year built
- And other relevant real estate market indicators.

## Technologies Used
- **Streamlit**: For building the interactive web application.
- **Scikit-learn and XGBoost**: For model training, evaluation, and predictions.
- **Pandas and NumPy**: For data preprocessing and manipulation.
- **Matplotlib and Seaborn**: For exploratory data analysis and visualization.

## Models Included
The predictive application includes various regression models such as:
- Linear Regression
- Decision Tree
- Random Forest
- XGBoost

These models have been trained, evaluated, and serialized for use within the application.

## Future Enhancements
- Incorporating additional datasets for regional analysis.
- Adding explainability tools such as SHAP for model interpretability.
- Expanding the application to include rental price predictions.

---

## Run Locally

1. **Clone this repo:**
```bash
git clone https://github.com/Fotinac/week10-real-estate-model.git
cd week10-real-estate-model
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run app_week10_load_model.py
```
---

## Author

- **Name**: Fotinacao  
- **Course**: CST2216  
- **Instructor**: Swapnil Kangralkar  
- **Institution**: Algonquin College

---

## License

This project is for educational purposes only.

