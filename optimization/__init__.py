"""
Advanced Grid Optimization Engine
Uses forecasts to suggest optimal grid operations
"""

import logging
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

logger = logging.getLogger("enerwise.grid_optimizer")

class GridOptimizer:
    """Advanced optimization for grid operations"""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> Dict:
        """Load grid optimization rules and constraints"""
        return {
            "peak_threshold": 1000,  # MW
            "ramp_threshold": 100,   # MW/h
            "reserve_margin": 0.15,  # 15%
            "congestion_limit": 950, # MW
            "voltage_stability": 0.95
        }
    
    async def optimize_grid_operations(self, 
                                    load_forecast: Dict,
                                    renewable_forecast: Dict,
                                    grid_state: Dict) -> Dict[str, Any]:
        """Main optimization function"""
        
        recommendations = []
        
        # 1. Load-resource balance optimization
        balance_recs = await self._optimize_load_balance(
            load_forecast, renewable_forecast, grid_state
        )
        recommendations.extend(balance_recs)
        
        # 2. Congestion management
        congestion_recs = await self._optimize_congestion_management(
            load_forecast, grid_state
        )
        recommendations.extend(congestion_recs)
        
        # 3. Reserve optimization
        reserve_recs = await self._optimize_reserves(
            load_forecast, grid_state
        )
        recommendations.extend(reserve_recs)
        
        # 4. Economic dispatch
        economic_recs = await self._optimize_economic_dispatch(
            load_forecast, grid_state
        )
        recommendations.extend(economic_recs)
        
        return {
            "recommendations": recommendations,
            "optimization_score": self._calculate_optimization_score(recommendations),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _optimize_load_balance(self, load_forecast: Dict, 
                                   renewable_forecast: Dict,
                                   grid_state: Dict) -> List[Dict]:
        """Optimize balance between load and available resources"""
        recommendations = []
        
        load_values = [pred["load_mw"] for pred in load_forecast["predictions"]]
        renewable_values = renewable_forecast.get("values", [0] * len(load_values))
        
        net_load = [load - renewable for load, renewable in zip(load_values, renewable_values)]
        
        peak_net_load = max(net_load)
        if peak_net_load > self.optimization_rules["peak_threshold"]:
            recommendations.append({
                "type": "GENERATION_DISPATCH",
                "priority": "HIGH",
                "action": f"Dispatch additional generation for {peak_net_load:.1f} MW peak",
                "details": f"Net load exceeds threshold by {peak_net_load - self.optimization_rules['peak_threshold']:.1f} MW",
                "confidence": 0.92
            })
        
        return recommendations
    
    async def _optimize_congestion_management(self, load_forecast: Dict,
                                            grid_state: Dict) -> List[Dict]:
        """Manage potential grid congestion"""
        recommendations = []
        
        load_values = [pred["load_mw"] for pred in load_forecast["predictions"]]
        
        # Check for congestion risks
        congestion_risk_hours = [
            i for i, load in enumerate(load_values) 
            if load > self.optimization_rules["congestion_limit"]
        ]
        
        if congestion_risk_hours:
            risk_percentage = (len(congestion_risk_hours) / len(load_values)) * 100
            recommendations.append({
                "type": "CONGESTION_MANAGEMENT",
                "priority": "MEDIUM",
                "action": f"Monitor {len(congestion_risk_hours)} potential congestion hours",
                "details": f"{risk_percentage:.1f}% of forecast period at risk",
                "confidence": 0.85
            })
        
        return recommendations
    
    async def _optimize_reserves(self, load_forecast: Dict,
                               grid_state: Dict) -> List[Dict]:
        """Optimize spinning and operating reserves"""
        recommendations = []
        
        load_values = [pred["load_mw"] for pred in load_forecast["predictions"]]
        peak_load = max(load_values)
        
        required_reserves = peak_load * self.optimization_rules["reserve_margin"]
        current_reserves = grid_state.get("available_reserves", 0)
        
        if current_reserves < required_reserves:
            reserve_deficit = required_reserves - current_reserves
            recommendations.append({
                "type": "RESERVE_OPTIMIZATION",
                "priority": "HIGH",
                "action": f"Increase reserves by {reserve_deficit:.1f} MW",
                "details": f"Required: {required_reserves:.1f} MW, Available: {current_reserves:.1f} MW",
                "confidence": 0.90
            })
        
        return recommendations
    
    async def _optimize_economic_dispatch(self, load_forecast: Dict,
                                        grid_state: Dict) -> List[Dict]:
        """Optimize economic dispatch of generators"""
        recommendations = []
        
        # Simple economic optimization based on load patterns
        load_values = [pred["load_mw"] for pred in load_forecast["predictions"]]
        avg_load = np.mean(load_values)
        
        if avg_load < 700:  # Low load period
            recommendations.append({
                "type": "ECONOMIC_DISPATCH",
                "priority": "LOW",
                "action": "Consider shutting down least efficient units",
                "details": f"Average load {avg_load:.1f} MW allows optimization",
                "confidence": 0.78
            })
        
        return recommendations
    
    def _calculate_optimization_score(self, recommendations: List[Dict]) -> float:
        """Calculate overall optimization score"""
        if not recommendations:
            return 0.0
        
        priority_weights = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
        total_score = 0.0
        max_score = 0.0
        
        for rec in recommendations:
            weight = priority_weights.get(rec["priority"], 0.3)
            confidence = rec.get("confidence", 0.5)
            total_score += weight * confidence
            max_score += weight
        
        return (total_score / max_score) if max_score > 0 else 0.0

# Global instance
grid_optimizer = GridOptimizer()
