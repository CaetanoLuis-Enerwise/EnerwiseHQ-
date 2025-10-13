from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class EnergyType(str, Enum):
    SOLAR = "solar"
    WIND = "wind"
    HYDRO = "hydro"
    DEMAND = "demand"
    SUPPLY = "supply"
    GRID = "grid"

class GridRegion(str, Enum):
    NORTH = "north_portugal"
    CENTER = "central_portugal" 
    SOUTH = "south_portugal"
    LISBON = "lisbon_grid"
    PORTO = "porto_grid"

class EnergyForecastRequest(BaseModel):
    region: GridRegion = Field(..., example=GridRegion.LISBON)
    energy_type: EnergyType = Field(..., example=EnergyType.DEMAND)
    values: List[float] = Field(..., example=[500.0, 520.0, 540.0, 560.0])
    horizon: int = Field(24, ge=1, le=168, example=24)
    include_confidence: bool = True
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

class EnergyForecastResponse(BaseModel):
    region: str
    energy_type: str
    forecast: List[float]
    confidence_intervals: Dict[str, Any]
    mode: str
    timestamp: str
    model_metadata: Dict[str, Any]
    summary: Dict[str, Any]
