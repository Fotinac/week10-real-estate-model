from sklearn.metrics import mean_squared_error, r2_score
from utils.logger import setup_logger

logger = setup_logger()

def evaluate_model(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logger.info(f"Model Evaluation - MSE: {mse}, R2: {r2}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
