import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_test, predictions):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, predictions, alpha=0.7, color='skyblue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs. Predicted Real Estate Prices")
    ax.legend()
    ax.grid(True)
    return fig

