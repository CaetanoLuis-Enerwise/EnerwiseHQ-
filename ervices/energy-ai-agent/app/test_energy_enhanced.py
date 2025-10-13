import asyncio
import requests
import json

async def test_enhanced_energy_api():
    print("🧪 Testing Enhanced Energy API...")
    
    # Test data for different energy types
    test_requests = [
        {
            "region": "lisbon_grid",
            "energy_type": "demand",
            "values": [500, 520, 540, 560, 580, 600, 620, 640],
            "horizon": 6,
            "include_confidence": True
        },
        {
            "region": "north_portugal", 
            "energy_type": "solar",
            "values": [100, 150, 200, 250, 300, 280, 220, 180],
            "horizon": 6,
            "include_confidence": True
        }
    ]
    
    base_url = "http://localhost:8000"
    
    for i, request_data in enumerate(test_requests, 1):
        print(f"\n🔍 Test {i}: {request_data['energy_type'].upper()} in {request_data['region']}")
        
        try:
            response = requests.post(
                f"{base_url}/api/energy-forecast",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Status: SUCCESS")
                print(f"   ✅ Forecast: {result['forecast'][:3]}...")
                print(f"   ✅ Mode: {result['mode']}")
                print(f"   ✅ Summary: {result['summary']}")
            else:
                print(f"   ❌ Status: FAILED ({response.status_code})")
                print(f"   ❌ Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    print("\n🎉 Enhanced energy API testing completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_energy_api())
