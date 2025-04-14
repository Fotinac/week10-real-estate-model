import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from utils.data_loader import load_data
from utils.preprocessing import preprocess_data
from utils.evaluator import evaluate_model
from utils.logger import setup_logger

logger = setup_logger()

def train_and_save_model(file_path, model_path='models/linear_regression_model.pkl'):
    try:
        df = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.info("Model trained.")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise
