from pathlib import Path

class Config:
    def __init__(self):
        self.horizon = 1
        self.model_mode = "light"
        self.timesteps = 24
        self.n_splits = 3
        self.min_consumo = 0.1
        
        # Paths
        self.consumo_dir = Path("data/consumo")
        self.pv_dir = Path("data/pv") 
        self.output_dir = Path("outputs")
        
        # Training params
        self.epochs_light = 10
        self.batch_size_light = 32
        
        # Model flags
        self.tf_available = True
        self.xgb_available = True  
        self.prophet_available = True

cfg = Config()
