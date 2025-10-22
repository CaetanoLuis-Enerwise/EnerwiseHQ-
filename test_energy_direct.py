import asyncio
import sys
import os
sys.path.append('.')

from models.energy_models import EnergyForecastRequest, EnergyType, GridRegion
from forecasting.advanced_forecaster import advanced_forecaster

async def test_direct():
    print("🧪 Testing Energy AI Directly...")
    
    test_requests = [
        EnergyForecastRequest(
            region=GridRegion.LISBON,
            energy_type=EnergyType.DEMAND,
            values=[500, 520, 540, 560, 580, 600, 620, 640],
            horizon=6
        ),
        EnergyForecastRequest(
            region=GridRegion.NORTH, 
            energy_type=EnergyType.SOLAR,
            values=[100, 150, 200, 250, 300, 280, 220, 180],
            horizon=6
        )
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n🔍 Test {i}: {request.energy_type.value} in {request.region.value}")
        
        result = await advanced_forecaster.forecast(request.values, request.horizon)
        
        print(f"   ✅ Forecast: {result['ensemble_forecast'][:3]}...")
        print(f"   ✅ Mode: {result['mode']}")
        print(f"   ✅ Confidence: {result['confidence_intervals']['confidence_score']}%")
    
    print("\n🎉 Direct testing completed!")

if __name__ == "__main__":
    asyncio.run(test_direct())