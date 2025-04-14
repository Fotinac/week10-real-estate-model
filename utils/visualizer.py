import matplotlib.pyplot as plt
from utils.logger import setup_logger

logger = setup_logger()

def plot_predictions(y_true, y_pred):
    try:
        plt.scatter(y_true, y_pred)
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs. Predicted Prices")
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red')
        plt.grid(True)
        plt.show()
        logger.info("Plot displayed.")
    except Exception as e:
        logger.error(f"Failed to plot predictions: {e}")
        raise
