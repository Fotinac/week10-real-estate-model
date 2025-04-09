from sklearn.tree import DecisionTreeRegressor

def train_model(X_train, y_train):
    """
    Train a Decision Tree Regressor.

    Returns:
    - Trained model
    """
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model
