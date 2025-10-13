from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime
from enum import Enum

class EnergyType(str, Enum):
    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro" 
    DEMAND = "demand"
    SUPPLY = "supply"
    GRID = "grid"

class ForecastRequest(BaseModel):
    region: str = Field(..., example="north_portugal")
    energy_type: EnergyType
    values: List[float] = Field(..., example=[500.0, 520.0, 540.0, 560.0])
    horizon: int = Field(24, ge=1, le=168, example=24)
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

class ConfidenceIntervals(BaseModel):
    lower_bound: List[float]
    upper_bound: List[float] 
    confidence_score: float

class ForecastResponse(BaseModel):
    region: str
    energy_type: EnergyType
    forecast: List[float]
    confidence: ConfidenceIntervals
    mode: str
    timestamp: str
