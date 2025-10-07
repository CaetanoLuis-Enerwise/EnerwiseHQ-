import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Importar do mesmo diretório
from models import ModelBuilder, Trainer, FeatureEngineer
from config import cfg
from utils import aggregate_phases, detect_outliers_zscore

# Logging
logger = logging.getLogger("enerwise_pipeline")
logger.setLevel(logging.INFO)

def run_pipeline(cfg, horizon: Optional[int] = None) -> pd.DataFrame:
    """
    Executa pipeline Super-Híbrido e retorna DataFrame com previsões.
    JSON-friendly para FastAPI.
    """
    start_time = time.time()
    logger.info("Iniciando pipeline Super-Híbrido...")

    # 1) Carregar dados (simulados para demo)
    consumo = aggregate_phases(cfg.consumo_dir, "Consumo")
    pv = aggregate_phases(cfg.pv_dir, "PV")
    
    if consumo.empty or pv.empty:
        # Criar dados demo se não existirem
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='30min')
        consumo = pd.DataFrame({
            'total_consumo': np.random.uniform(5, 15, len(dates))
        }, index=dates)
        pv = pd.DataFrame({
            'total_pv': np.random.uniform(0, 8, len(dates))
        }, index=dates)

    df = consumo.join(pv, how="outer")
    df.columns = [c.lower() for c in df.columns]

    # Ajustar colunas
    df["total_consumo"] = df.get("total_consumo", df.iloc[:,0])
    df["total_pv"] = df.get("total_pv", df.iloc[:,1] if df.shape[1]>1 else 0)
    
    df["consumo_liquido"] = df["total_consumo"] - df["total_pv"]
    df["excesso_pv"] = (df["total_pv"] - df["total_consumo"]).clip(lower=0.0)

    # 2) Corrigir outliers
    mask_anomalies, n_outliers = detect_outliers_zscore(df, "consumo_liquido", threshold=4.0)
    if mask_anomalies.any():
        df.loc[mask_anomalies, "consumo_liquido"] = df["consumo_liquido"].interpolate(method="linear", limit_direction="both")
        logger.info(f"Corrigidos {n_outliers} outliers")

    # 3) Feature engineering
    fe = FeatureEngineer(cfg.min_consumo)
    df_feat = fe.create_features(df[["consumo_liquido", "total_pv", "excesso_pv"]])

    # 4) Preparar dados sequenciais
    trainer = Trainer(cfg)
    series_values = df_feat["consumo_liquido"].fillna(method='ffill').values.reshape(-1, 1)
    
    if len(series_values) > 1:
        trainer.scaler_seq.fit(series_values)
        series_scaled = trainer.scaler_seq.transform(series_values).flatten()
    else:
        series_scaled = series_values.flatten()

    horizon_used = cfg.horizon if horizon is None else horizon
    
    # Criar sequências se dados suficientes
    if len(series_scaled) > cfg.timesteps + horizon_used:
        X_seq_all, y_seq_all = trainer.create_sequences(series_scaled, cfg.timesteps, horizon_used)
    else:
        # Fallback: dados insuficientes
        logger.warning("Dados insuficientes para sequências, usando fallback")
        fake_pred = np.array([df_feat["consumo_liquido"].mean()] * min(50, len(df_feat)))
        out_df = pd.DataFrame({
            "LSTM_mean": fake_pred,
            "Transformer_mean": fake_pred,
            "XGBoost_mean": fake_pred,
            "Prophet_mean": fake_pred
        }, index=df_feat.index[-len(fake_pred):])
        return out_df

    # 5) Cross-validation + Model training
    tscv = TimeSeriesSplit(n_splits=min(cfg.n_splits, len(X_seq_all)-1))
    mb = ModelBuilder(cfg)
    all_preds_per_model: Dict[str, List[np.ndarray]] = {}
    
    models_available = {
        "LSTM": cfg.tf_available,
        "Transformer": cfg.tf_available,
        "XGBoost": cfg.xgb_available,
        "Prophet": cfg.prophet_available
    }

    logger.info(f"Treinando com {tscv.get_n_splits()} folds")
    
    for fold_no, (train_idx, test_idx) in enumerate(tscv.split(X_seq_all), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
            
        X_train_seq, X_test_seq = X_seq_all[train_idx], X_seq_all[test_idx]
        y_train_seq, y_test_seq = y_seq_all[train_idx], y_seq_all[test_idx]

        fold_predictions = {}

        # LSTM
        if models_available["LSTM"]:
            try:
                model_lstm = mb.build_lstm(cfg.timesteps, horizon_used)
                if model_lstm is not None:
                    model_lstm.fit(X_train_seq, y_train_seq, epochs=cfg.epochs_light, 
                                 batch_size=cfg.batch_size_light, verbose=0)
                    preds_lstm = trainer.inverse_transform_seq(model_lstm.predict(X_test_seq))
                    fold_predictions["LSTM"] = preds_lstm
            except Exception as e:
                logger.warning(f"LSTM fold {fold_no} falhou: {e}")

        # Transformer
        if models_available["Transformer"]:
            try:
                model_tr = mb.build_transformer(cfg.timesteps, horizon_used)
                if model_tr is not None:
                    model_tr.fit(X_train_seq, y_train_seq, epochs=cfg.epochs_light,
                               batch_size=cfg.batch_size_light, verbose=0)
                    preds_tr = trainer.inverse_transform_seq(model_tr.predict(X_test_seq))
                    fold_predictions["Transformer"] = preds_tr
            except Exception as e:
                logger.warning(f"Transformer fold {fold_no} falhou: {e}")

        # XGBoost
        if models_available["XGBoost"]:
            try:
                model_xgb = mb.build_xgboost()
                if model_xgb is not None:
                    X_tab_train, X_tab_test, y_tab_train, y_tab_test = trainer.prepare_tabular_data(df_feat, train_idx, test_idx)
                    model_xgb.fit(X_tab_train, y_tab_train)
                    preds_xgb = model_xgb.predict(X_tab_test)
                    fold_predictions["XGBoost"] = np.repeat(preds_xgb.reshape(-1, 1), horizon_used, axis=1)
            except Exception as e:
                logger.warning(f"XGBoost fold {fold_no} falhou: {e}")

        # Prophet
        if models_available["Prophet"]:
            try:
                model_prop = mb.build_prophet()
                if model_prop is not None:
                    df_prop_train, df_prop_test = trainer.prepare_prophet(df_feat, train_idx, test_idx)
                    if len(df_prop_train) > 10:
                        model_prop.fit(df_prop_train)
                        future = df_prop_test[["ds"]]
                        forecast = model_prop.predict(future)
                        preds_prop = np.repeat(forecast["yhat"].values.reshape(-1, 1), horizon_used, axis=1)
                        fold_predictions["Prophet"] = preds_prop
            except Exception as e:
                logger.warning(f"Prophet fold {fold_no} falhou: {e}")

        # Salvar previsões
        for k, v in fold_predictions.items():
            all_preds_per_model.setdefault(k, []).append(v)

    # 6) Agregar previsões médias
    out_data = {}
    for model_name, preds_list in all_preds_per_model.items():
        if preds_list:
            min_len = min([p.shape[0] for p in preds_list])
            preds_trim = np.array([p[:min_len] for p in preds_list])
            mean_pred = preds_trim.mean(axis=0).mean(axis=1)
            out_data[f"{model_name}_mean"] = mean_pred

    # Criar DataFrame de saída
    if out_data:
        out_df = pd.DataFrame(out_data, index=df_feat.index[-len(mean_pred):])
    else:
        # Fallback se nenhum modelo funcionou
        logger.warning("Nenhum modelo produziu previsões, usando fallback")
        fallback_pred = np.array([df_feat["consumo_liquido"].mean()] * 50)
        out_df = pd.DataFrame({"Fallback_mean": fallback_pred}, 
                            index=df_feat.index[-50:])

    elapsed = time.time() - start_time
    logger.info(f"Pipeline concluído em {elapsed:.2f}s")
    return out_df
