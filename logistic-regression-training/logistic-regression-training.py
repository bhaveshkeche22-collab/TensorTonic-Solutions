import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 
                    1/(1+np.exp(-z)), 
                    np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    m, n = X.shape
    
    # Initialize weights and bias
    w = np.zeros(n)
    b = 0.0
    
    for _ in range(steps):
        
        # Linear combination
        z = np.dot(X, w) + b
        
        # Apply sigmoid
        y_hat = _sigmoid(z)
        
        # Compute gradients
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
    
    return w, b