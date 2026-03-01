import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values using column mean or median.
    If a column is all NaN, fill with 0.
    Works for both 1D and 2D inputs.
    """
    X = np.array(X, dtype=float)

    # Handle 1D case separately
    if X.ndim == 1:
        if strategy == 'mean':
            value = np.nanmean(X)
        elif strategy == 'median':
            value = np.nanmedian(X)
        else:
            raise ValueError("strategy must be 'mean' or 'median'")
        
        # If all values were NaN → replace with 0
        if np.isnan(value):
            value = 0.0
        
        X[np.isnan(X)] = value
        return X

    # 2D case
    if strategy == 'mean':
        values = np.nanmean(X, axis=0)
    elif strategy == 'median':
        values = np.nanmedian(X, axis=0)
    else:
        raise ValueError("strategy must be 'mean' or 'median'")

    # Replace NaN column stats (all-NaN columns) with 0
    values = np.where(np.isnan(values), 0.0, values)

    nan_rows, nan_cols = np.where(np.isnan(X))
    X[nan_rows, nan_cols] = values[nan_cols]

    return X