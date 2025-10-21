from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import pandas as pd
import uvicorn
import logging

# Importar do mesmo diretório
from pipeline import run_pipeline
from config import cfg

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enerwise_api")

app = FastAPI(
    title="Enerwise AI Forecast Service",
    description="Serviço de previsão de consumo energético com IA",
    version="1.0.0"
)

class ForecastRequest(BaseModel):
    horizon: Optional[int] = 1  # Previsão em passos de 30min
    mode: Optional[str] = "light"  # "light" ou "normal"

class ForecastResponse(BaseModel):
    status: str
    message: str
    predictions: list
    horizon: int

@app.get("/")
def read_root():
    return {"message": "Enerwise AI Forecast Service está rodando!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "enerwise-ai"}

@app.post("/forecast/", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """
    Endpoint para previsão de consumo energético
    """
    try:
        logger.info(f"Recebida requisição: horizon={req.horizon}, mode={req.mode}")
        
        # Atualizar configuração
        cfg.horizon = req.horizon
        cfg.model_mode = req.mode
        
        # Executar pipeline
        predictions_df = run_pipeline(cfg, horizon=req.horizon)
        
        if predictions_df.empty:
            raise HTTPException(
                status_code=500, 
                detail="Pipeline não gerou previsões válidas"
            )
        
        # Converter para formato JSON
        predictions_json = []
        for idx, row in predictions_df.iterrows():
            pred_record = {"timestamp": idx.isoformat()}
            for col in predictions_df.columns:
                pred_record[col] = float(row[col])
            predictions_json.append(pred_record)
        
        logger.info(f"Previsão gerada com {len(predictions_json)} registros")
        
        return ForecastResponse(
            status="success",
            message="Previsão gerada com sucesso",
            predictions=predictions_json,
            horizon=req.horizon
        )
        
    except Exception as e:
        logger.error(f"Erro no pipeline: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno no servidor: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
