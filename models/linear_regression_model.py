from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    """
    Train a Linear Regression model.

    Parameters:
    - X_train: training features
    - y_train: training target

    Returns:
    - Trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
