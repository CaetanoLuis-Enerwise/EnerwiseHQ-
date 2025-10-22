import asyncio
import os
import sys
sys.path.append('.')

# Test simple import first
try:
    from forecasting.advanced_forecaster import advanced_forecaster
    print("✅ Advanced forecaster imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test():
    print("🧪 Testing Advanced Forecaster...")
    
    # Test data
    test_data = [500, 520, 540, 560, 580, 600, 620, 640, 620, 600, 580, 560]
    
    # Test 1: Built-in mode (default)
    print("1. Testing BUILT-IN mode...")
    result1 = await advanced_forecaster.forecast(test_data, horizon=6)
    print(f"   ✅ Mode: {result1['mode']}")
    print(f"   ✅ Forecast: {result1['ensemble_forecast'][:3]}...")  # Show first 3 values
    print(f"   ✅ Confidence: {result1['confidence_intervals']['confidence_score']}%")
    
    print("🎉 Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test())