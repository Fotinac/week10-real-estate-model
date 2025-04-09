# from models.linear_regression_model import train_model
# from models.decision_tree_model import train_model
# from models.random_forest_model import train_model
# from models.xgboost_model import train_model

import logging

# === Import your custom modules ===
from utils.data_loader import load_data
from utils.preprocessing import split_data
from utils.evaluator import evaluate_model
from utils.visualizer import plot_actual_vs_predicted

# Choose ONE model to import here:
from models.linear_regression_model import train_model
# from models.decision_tree_model import train_model
# from models.random_forest_model import train_model
# from models.xgboost_model import train_model

# === Set up logging ===
logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # 1. Load data
        data_path = "data/final.csv"
        df = load_data(data_path)
        logging.info("Data loaded successfully.")
        
        # 2. Split data
        X_train, X_test, y_train, y_test = split_data(df, target_column="price")
        logging.info("Data split into training and testing sets.")

        # 3. Train model
        model = train_model(X_train, y_train)
        logging.info(f"Model trained: {model.__class__.__name__}")

        # 4. Evaluate model
        mse, r2 = evaluate_model(model, X_test, y_test)
        print(f"Model Performance:\nMSE: {mse:.2f}\nR²: {r2:.2f}")
        logging.info(f"Evaluation done. MSE: {mse:.2f}, R²: {r2:.2f}")

        # 5. Visualize
        predictions = model.predict(X_test)
        plot_actual_vs_predicted(y_test, predictions)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print("Something went wrong. Check logs/app.log for details.")

if __name__ == "__main__":
    main()
