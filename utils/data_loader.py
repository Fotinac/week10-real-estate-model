import pandas as pd
from utils.logger import setup_logger

logger = setup_logger()

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
