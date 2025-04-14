from sklearn.model_selection import train_test_split
from utils.logger import setup_logger

logger = setup_logger()

def preprocess_data(df):
    try:
        X = df[['Area', 'Bedrooms', 'Bathrooms']]
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Data preprocessed and split into train/test sets.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
