from sklearn.model_selection import train_test_split

def split_data(df, target_column):
    """
    Split data into training and testing sets.
    
    Parameters:
    - df: pandas DataFrame
    - target_column: name of the column to predict

    Returns:
    - X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
