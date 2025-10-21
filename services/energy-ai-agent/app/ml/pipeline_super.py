import numpy as np

def infer_from_series(historical_data, horizon=24, features=None):
    """
    Your university research pipeline will go here later
    For now, just a simple placeholder
    """
    base = np.mean(historical_data[-24:]) if len(historical_data) > 24 else np.mean(historical_data)
    forecast = [base] * horizon
    confidence = {
        "lower_bound": [x * 0.95 for x in forecast],
        "upper_bound": [x * 1.05 for x in forecast],
        "confidence_score": 85.0
    }
    return {"forecast": forecast, "confidence": confidence}
