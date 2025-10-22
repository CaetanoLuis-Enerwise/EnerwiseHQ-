"""
Enerwise AI Energy Agent - Grid Operations AI
Core service for energy forecasting and grid optimization
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enerwise-ai-agent")

app = FastAPI(
    title="Enerwise AI Energy Agent",
    description="AI-powered grid operations and forecasting system",
    version="1.0.0"
)

class ForecastRequest(BaseModel):
    region: str
    horizon_hours: int = 24
    include_weather: bool = True

class GridAction(BaseModel):
    action_type: str
    description: str
    priority: str
    confidence: float

@app.get("/")
async def root():
    return {"message": "Enerwise AI Energy Agent 🧠⚡"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "energy-ai-agent"}

@app.post("/forecast/grid-load")
async def forecast_grid_load(request: ForecastRequest):
    """
    Main forecasting endpoint for grid operations
    """
    logger.info(f"Forecasting grid load for {request.region}")
    
    # TODO: Implement your enhanced forecasting models here
    return {
        "region": request.region,
        "forecast_horizon": request.horizon_hours,
        "peak_load": 1250.5,
        "peak_time": "2024-01-15T18:00:00",
        "confidence": 0.87,
        "notes": "Enhanced forecasting based on university research"
    }

@app.post("/optimize/grid-operations")
async def optimize_grid_operations(region: str):
    """
    Suggests grid optimization actions
    """
    return {
        "region": region,
        "suggested_actions": [
            {
                "action_type": "GENERATION_ADJUSTMENT",
                "description": "Increase spinning reserve by 50MW",
                "priority": "MEDIUM",
                "confidence": 0.82
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
