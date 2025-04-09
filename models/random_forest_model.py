from sklearn.ensemble import RandomForestRegressor

def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor.

    Returns:
    - Trained model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
