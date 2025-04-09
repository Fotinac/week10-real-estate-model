from xgboost import XGBRegressor

def train_model(X_train, y_train):
    """
    Train an XGBoost Regressor.

    Returns:
    - Trained model
    """
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model
