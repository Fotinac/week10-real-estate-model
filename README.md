
# Week 10 Real Estate Price Prediction App

This project is part of **CST2216: Individual Term Project** at Algonquin College.  
It builds a modularized machine learning pipeline to predict real estate prices using multiple regression models. The app is deployed using Streamlit Cloud.

---

## Live App

[Click here to try the app](https://fotinac-week10-real-estate-model.streamlit.app)

---

## Project Features

- Load and preview real estate datasets
- Train and evaluate models such as:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor *(optional)*
- Evaluate model performance (MSE, R²)
- Visualize predictions vs actual prices
- Deploy and run via Streamlit Cloud

---

## Folder Structure

```
week10-real-estate-model/
├── app.py                  ← Streamlit app script
├── main.py                 ← Standalone script for training & testing
├── requirements.txt        ← Project dependencies
├── README.md               ← This file
├── data/                   ← CSV dataset(s)
├── models/                 ← ML model scripts (Linear, Tree, etc.)
├── utils/                  ← Helpers (data loading, preprocessing, etc.)
├── logs/                   ← Log files
```

---

## Run Locally

1. Clone this repository:
```bash
git clone https://github.com/Fotinac/week10-real-estate-model.git
cd week10-real-estate-model
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## Dependencies

Main packages used:
- `pandas`
- `scikit-learn`
- `matplotlib`
- `streamlit`

*See `requirements.txt` for full details.*

---

## ✍️ Author

- **Name**: Fotinacao
- **Course**: CST2216 — Business Intelligence System Infrastructure
- **Institution**: AC
- **Instructor**: Swapnil Kangralkar

---

## License

This project is for academic purposes only.
