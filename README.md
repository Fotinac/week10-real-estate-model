
## Live App  
[Click to open the app](https://week10-real-estate-model-d34ydl8cjghghgjwgc4d44.streamlit.app) 

---

# Real Estate Price Prediction App

This is a Week 10 project for **CST2216: Business Intelligence System Infrastructure** at Algonquin College.  
The app predicts real estate prices based on user inputs and a pre-trained linear regression model.

---

## Project Features

- Clean and intuitive Streamlit user interface
- Real-time user input for key housing features
- Pre-trained Linear Regression model for fast prediction
- Visual bar chart of predicted price
- Fully modular and deployment-ready

---

## Folder Structure

```
week10_real_estate_model/
├── app_week10_load_model.py     ← Streamlit app (loads pre-trained model)
├── train_and_save_model.py      ← Script to train and save model (optional)
├── models/
│   └── real_estate_model.pkl    ← Pre-trained model
├── data/
│   └── final.csv                ← Dataset used for training
├── requirements.txt             ← Python dependencies
├── README.md                    ← This file
```

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

## Features Used for Prediction

- `year_sold`
- `property_tax`
- `insurance`
- `beds`, `baths`, `sqft`, `year_built`, `lot_size`, `property_age`
- `basement`, `popular`, `recession`
- `property_type_Bunglow`, `property_type_Condo`

---

## Author

- **Name**: Fotinacao  
- **Course**: CST2216  
- **Instructor**: Swapnil Kangralkar  
- **Institution**: Algonquin College

---

## License

This project is for educational purposes only.

