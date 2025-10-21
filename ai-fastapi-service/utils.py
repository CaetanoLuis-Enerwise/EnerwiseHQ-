import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_phases(data_dir: Path, prefix: str) -> pd.DataFrame:
    """
    Agrega dados de múltiplas fases em um único DataFrame
    """
    try:
        # Simulação - na prática lería CSVs reais
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='30min')
        data = np.random.uniform(1, 10, len(dates))
        return pd.DataFrame({f'total_{prefix.lower()}': data}, index=dates)
    except Exception as e:
        print(f"Erro ao carregar dados {prefix}: {e}")
        return pd.DataFrame()

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0):
    """
    Detecta outliers usando Z-Score
    """
    if column not in df.columns:
        return np.array([False] * len(df)), 0
    
    series = df[column].dropna()
    if len(series) == 0:
        return np.array([False] * len(df)), 0
        
    z_scores = np.abs((series - series.mean()) / series.std())
    outliers_mask = z_scores > threshold
    
    # Criar mask com mesmo tamanho do DataFrame original
    full_mask = np.array([False] * len(df))
    valid_indices = df[column].notna()
    full_mask[valid_indices] = outliers_mask
    
    return full_mask, outliers_mask.sum()
