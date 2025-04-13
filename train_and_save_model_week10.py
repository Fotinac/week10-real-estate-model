import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

df = pd.read_csv("data/final.csv")
X = df.drop(columns=["price"])
y = df["price"]

model = LinearRegression()
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/real_estate_model.pkl")
