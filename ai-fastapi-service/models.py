import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, min_consumo: float = 0.1):
        self.min_consumo = min_consumo
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features temporais e de engenharia"""
        df = df.copy()
        
        # Features temporais
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek >= 5
        
        # Features de consumo
        if 'consumo_liquido' in df.columns:
            df['consumo_positivo'] = df['consumo_liquido'].clip(lower=self.min_consumo)
            df['consumo_log'] = np.log1p(df['consumo_positivo'])
            
        return df

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler_seq = RobustScaler()
        self.scaler_tab = RobustScaler()
    
    def create_sequences(self, data: np.ndarray, timesteps: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cria sequências para modelos temporais"""
        X, y = [], []
        for i in range(len(data) - timesteps - horizon + 1):
            X.append(data[i:(i + timesteps)])
            y.append(data[(i + timesteps):(i + timesteps + horizon)])
        return np.array(X), np.array(y)
    
    def inverse_transform_seq(self, data: np.ndarray) -> np.ndarray:
        """Reverte scaling das sequências"""
        return self.scaler_seq.inverse_transform(data.reshape(-1, 1)).flatten()
    
    def prepare_tabular_data(self, df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
        """Prepara dados para modelos tabulares"""
        features = [col for col in df.columns if col not in ['consumo_liquido', 'total_pv', 'excesso_pv']]
        X = df[features].values
        y = df['consumo_liquido'].values
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        return X_train, X_test, y_train, y_test
    
    def prepare_prophet(self, df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
        """Prepara dados para Prophet"""
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['consumo_liquido'].fillna(method='ffill').values
        })
        return prophet_df.iloc[train_idx], prophet_df.iloc[test_idx]

class ModelBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def build_lstm(self, timesteps: int, horizon: int):
        """Constrói modelo LSTM simplificado"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(timesteps, 1)),
                Dense(horizon)
            ])
            model.compile(optimizer='adam', loss='mse')
            return model
        except ImportError:
            logger.warning("TensorFlow não disponível para LSTM")
            return None
    
    def build_transformer(self, timesteps: int, horizon: int):
        """Constrói modelo Transformer simplificado"""
        try:
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Flatten
            
            inputs = Input(shape=(timesteps, 1))
            attention = MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
            normalized = LayerNormalization()(attention + inputs)
            flattened = Flatten()(normalized)
            outputs = Dense(horizon)(flattened)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='mse')
            return model
        except ImportError:
            logger.warning("TensorFlow não disponível para Transformer")
            return None
    
    def build_xgboost(self):
        """Constrói modelo XGBoost"""
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
        except ImportError:
            logger.warning("XGBoost não disponível")
            return None
    
    def build_prophet(self):
        """Constrói modelo Prophet"""
        try:
            from prophet import Prophet
            return Prophet(daily_seasonality=True, weekly_seasonality=True)
        except ImportError:
            logger.warning("Prophet não disponível")
            return None
