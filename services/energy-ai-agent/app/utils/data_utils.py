"""
Data Utilities for Enerwise Grid AI
Advanced data processing, validation, and transformation
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger("enerwise.data_utils")

class DataValidator:
    """Advanced data validation and quality assurance"""
    
    @staticmethod
    def validate_forecast_data(forecast_data: Dict) -> Tuple[bool, List[str]]:
        """Validate forecast data structure and values"""
        errors = []
        
        # Check required fields
        required_fields = ["predictions", "forecast_generated_at"]
        for field in required_fields:
            if field not in forecast_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate predictions array
        if "predictions" in forecast_data:
            predictions = forecast_data["predictions"]
            if not isinstance(predictions, list):
                errors.append("Predictions must be a list")
            else:
                for i, pred in enumerate(predictions):
                    if not isinstance(pred, dict):
                        errors.append(f"Prediction {i} must be a dictionary")
                    else:
                        if "timestamp" not in pred:
                            errors.append(f"Prediction {i} missing timestamp")
                        if "load_mw" not in pred:
                            errors.append(f"Prediction {i} missing load_mw")
        
        # Validate numerical ranges
        if "predictions" in forecast_data:
            for i, pred in enumerate(forecast_data["predictions"]):
                if "load_mw" in pred:
                    load = pred["load_mw"]
                    if not isinstance(load, (int, float)) or load < 0 or load > 10000:
                        errors.append(f"Invalid load value at prediction {i}: {load}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def detect_anomalies(data_series: List[float], 
                        method: str = "iqr") -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        if not data_series:
            return {"anomalies": [], "thresholds": {}}
        
        series = np.array(data_series)
        anomalies = []
        
        if method == "iqr":
            # Interquartile Range method
            Q1 = np.percentile(series, 25)
            Q3 = np.percentile(series, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            for i, value in enumerate(series):
                if value < lower_bound or value > upper_bound:
                    anomalies.append({
                        "index": i,
                        "value": value,
                        "reason": "outside_iqr_bounds",
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    })
        
        elif method == "zscore":
            # Z-score method
            mean = np.mean(series)
            std = np.std(series)
            z_scores = np.abs((series - mean) / std)
            
            for i, z_score in enumerate(z_scores):
                if z_score > 3:  # 3 standard deviations
                    anomalies.append({
                        "index": i,
                        "value": series[i],
                        "z_score": z_score,
                        "reason": "high_z_score"
                    })
        
        return {
            "anomalies": anomalies,
            "method": method,
            "total_detected": len(anomalies),
            "data_points": len(series)
        }

class DataTransformer:
    """Advanced data transformation and feature engineering"""
    
    @staticmethod
    def create_time_features(timestamps: List[str]) -> Dict[str, List[float]]:
        """Create comprehensive time-based features"""
        features = {
            "hour_sin": [],
            "hour_cos": [],
            "day_sin": [],
            "day_cos": [],
            "month_sin": [],
            "month_cos": [],
            "is_weekend": [],
            "is_holiday": []  # Would need holiday calendar
        }
        
        for ts in timestamps:
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                
                # Cyclical hour encoding
                hour_rad = 2 * np.pi * dt.hour / 24
                features["hour_sin"].append(np.sin(hour_rad))
                features["hour_cos"].append(np.cos(hour_rad))
                
                # Cyclical day of year encoding
                day_of_year = dt.timetuple().tm_yday
                day_rad = 2 * np.pi * day_of_year / 365
                features["day_sin"].append(np.sin(day_rad))
                features["day_cos"].append(np.cos(day_rad))
                
                # Cyclical month encoding
                month_rad = 2 * np.pi * (dt.month - 1) / 12
                features["month_sin"].append(np.sin(month_rad))
                features["month_cos"].append(np.cos(month_rad))
                
                # Weekend flag
                features["is_weekend"].append(1.0 if dt.weekday() >= 5 else 0.0)
                
                # Holiday flag (simplified)
                features["is_holiday"].append(0.0)  # Would use holiday calendar
                
            except Exception as e:
                logger.warning(f"Error processing timestamp {ts}: {e}")
                # Fill with neutral values
                for key in features:
                    features[key].append(0.0)
        
        return features
    
    @staticmethod
    def calculate_technical_indicators(data_series: List[float], 
                                     window_sizes: List[int] = [3, 6, 12, 24]) -> Dict[str, List[float]]:
        """Calculate technical indicators for time series"""
        series = np.array(data_series)
        indicators = {}
        
        # Moving averages
        for window in window_sizes:
            key = f"ma_{window}"
            indicators[key] = DataTransformer._moving_average(series, window)
        
        # Exponential moving averages
        for window in [6, 12, 24]:
            key = f"ema_{window}"
            indicators[key] = DataTransformer._exponential_moving_average(series, window)
        
        # Rate of Change
        indicators["roc_1"] = DataTransformer._rate_of_change(series, 1)
        indicators["roc_6"] = DataTransformer._rate_of_change(series, 6)
        
        # Volatility
        for window in [6, 12]:
            key = f"volatility_{window}"
            indicators[key] = DataTransformer._rolling_volatility(series, window)
        
        return indicators
    
    @staticmethod
    def _moving_average(series: np.ndarray, window: int) -> List[float]:
        """Calculate simple moving average"""
        if len(series) < window:
            return [np.nan] * len(series)
        return [np.nan] * (window - 1) + \
               [np.mean(series[i-window+1:i+1]) for i in range(window-1, len(series))]
    
    @staticmethod
    def _exponential_moving_average(series: np.ndarray, window: int) -> List[float]:
        """Calculate exponential moving average"""
        if len(series) < window:
            return [np.nan] * len(series)
        
        alpha = 2 / (window + 1)
        ema = [series[0]]
        for i in range(1, len(series)):
            ema.append(alpha * series[i] + (1 - alpha) * ema[i-1])
        return ema
    
    @staticmethod
    def _rate_of_change(series: np.ndarray, period: int) -> List[float]:
        """Calculate rate of change"""
        if len(series) <= period:
            return [0.0] * len(series)
        
        roc = [0.0] * period
        for i in range(period, len(series)):
            if series[i - period] != 0:
                change = (series[i] - series[i - period]) / series[i - period] * 100
            else:
                change = 0.0
            roc.append(change)
        return roc
    
    @staticmethod
    def _rolling_volatility(series: np.ndarray, window: int) -> List[float]:
        """Calculate rolling volatility (standard deviation)"""
        if len(series) < window:
            return [0.0] * len(series)
        
        volatility = [0.0] * (window - 1)
        for i in range(window - 1, len(series)):
            window_data = series[i-window+1:i+1]
            volatility.append(np.std(window_data))
        return volatility

class PerformanceMetrics:
    """Advanced performance metrics for model evaluation"""
    
    @staticmethod
    def calculate_forecast_metrics(actual: List[float], 
                                 predicted: List[float]) -> Dict[str, float]:
        """Calculate comprehensive forecast accuracy metrics"""
        if len(actual) != len(predicted) or not actual:
            return {"error": "Invalid input data"}
        
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Basic metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        
        # Percentage errors
        mape = np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-6))) * 100
        smape = 100 * np.mean(2 * np.abs(predicted - actual) / 
                             (np.abs(actual) + np.abs(predicted) + 1e-6))
        
        # Correlation
        if len(actual) > 1:
            correlation = np.corrcoef(actual, predicted)[0, 1]
        else:
            correlation = 0.0
        
        # Bias
        bias = np.mean(predicted - actual)
        
        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "smape": round(smape, 2),
            "correlation": round(correlation, 4),
            "bias": round(bias, 4),
            "mean_actual": round(np.mean(actual), 2),
            "mean_predicted": round(np.mean(predicted), 2)
        }
    
    @staticmethod
    def calculate_confidence_intervals(predictions: List[float], 
                                     confidence: float = 0.95) -> Dict[str, float]:
        """Calculate prediction confidence intervals"""
        if not predictions:
            return {"lower": 0.0, "upper": 0.0}
        
        pred_array = np.array(predictions)
        mean = np.mean(pred_array)
        std = np.std(pred_array)
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        margin = z_score * std
        
        return {
            "lower": round(mean - margin, 2),
            "upper": round(mean + margin, 2),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "confidence_level": confidence
        }
