import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    w = np.linalg.pinv(X.T @ X) @ X.T @ y

    return w