

##  ENERWISE FORECASTING  — COMPLETE SYSTEM

---

### `main_pipeline.py`

```python
import os
import yaml
from core.data_loader import load_energy_data
from core.feature_engineering import engineer_features
from core.model_lstm import train_lstm, forecast_lstm
from core.model_prophet import train_prophet, forecast_prophet
from core.model_xgboost import train_xgb, forecast_xgb
from core.model_transformer import train_transformer, forecast_transformer
from core.reporter import generate_report
from datetime import datetime

def main():
    # Load config
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    print(" Enerwise Forecasting Core — Initialization")

    # Load and preprocess data
    df = load_energy_data(config["data"]["path"])
    df = engineer_features(df)
    print(" Data loaded and features engineered")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Create output directories
    os.makedirs("outputs/forecasts", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    results = {}

    # Run LSTM
    if config["models"]["lstm"]:
        model_lstm = train_lstm(df)
        results["LSTM"] = forecast_lstm(model_lstm, df)
        print(" LSTM forecast complete")

    # Run Prophet
    if config["models"]["prophet"]:
        model_prophet = train_prophet(df)
        results["Prophet"] = forecast_prophet(model_prophet, df)
        print(" Prophet forecast complete")

    # Run XGBoost
    if config["models"]["xgboost"]:
        model_xgb = train_xgb(df)
        results["XGBoost"] = forecast_xgb(model_xgb, df)
        print(" XGBoost forecast complete")

    # Run Transformer
    if config["models"]["transformer"]:
        model_trans = train_transformer(df)
        results["Transformer"] = forecast_transformer(model_trans, df)
        print(" Transformer forecast complete")

    # Generate report
    generate_report(results, timestamp)
    print(" Forecast report generated")

    print(" Enerwise Forecasting Core completed successfully")

if __name__ == "__main__":
    main()
```

---

### `configs/config.yaml`

```yaml
data:
  path: "data/energy_consumption.csv"

models:
  lstm: true
  prophet: true
  xgboost: true
  transformer: true

forecast:
  horizon: 72
  frequency: "H"

report:
  format: "pdf"
```

---

### `core/data_loader.py`

```python
import pandas as pd

def load_energy_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.dropna()
    return df
```

---

### `core/feature_engineering.py`

```python
import pandas as pd

def engineer_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["weekday"] = df["timestamp"].dt.weekday
    df["rolling_mean_24h"] = df["consumption"].rolling(24, min_periods=1).mean()
    df["rolling_std_24h"] = df["consumption"].rolling(24, min_periods=1).std()
    df = df.dropna()
    return df
```

---

### `core/model_lstm.py`

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm(df):
    data = df["consumption"].values
    seq_length = 24
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    X, y = np.array(X), np.array(y)
    X = np.expand_dims(X, axis=-1)

    model = Sequential([
        LSTM(64, input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def forecast_lstm(model, df):
    last_sequence = df["consumption"].values[-24:]
    preds = []
    seq = last_sequence.copy()
    for _ in range(72):
        x_input = np.expand_dims(seq[-24:], axis=(0, -1))
        pred = model.predict(x_input, verbose=0)[0][0]
        preds.append(pred)
        seq = np.append(seq, pred)
    return preds
```

---

### `core/model_prophet.py`

```python
from prophet import Prophet
import pandas as pd

def train_prophet(df):
    prophet_df = df.rename(columns={"timestamp": "ds", "consumption": "y"})
    model = Prophet()
    model.fit(prophet_df)
    return model

def forecast_prophet(model, df):
    future = model.make_future_dataframe(periods=72, freq="H")
    forecast = model.predict(future)
    return forecast.tail(72)["yhat"].values.tolist()
```

---

### `core/model_xgboost.py`

```python
import xgboost as xgb
import pandas as pd

def train_xgb(df):
    features = ["hour", "day", "month", "weekday", "rolling_mean_24h", "rolling_std_24h"]
    X = df[features]
    y = df["consumption"]
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
    model.fit(X, y)
    return model

def forecast_xgb(model, df):
    last_rows = df.tail(72)
    features = ["hour", "day", "month", "weekday", "rolling_mean_24h", "rolling_std_24h"]
    preds = model.predict(last_rows[features])
    return preds.tolist()
```

---

### `core/model_transformer.py`

```python
import torch
import torch.nn as nn
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x[:, -1, :])

def train_transformer(df):
    data = df["consumption"].values
    seq_len = 24
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    X, y = torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

    model = TransformerModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()

    return model

def forecast_transformer(model, df):
    seq = torch.tensor(df["consumption"].values[-24:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    preds = []
    with torch.no_grad():
        for _ in range(72):
            pred = model(seq).item()
            preds.append(pred)
            new_seq = torch.cat([seq[:, 1:, :], torch.tensor([[[pred]]])], dim=1)
            seq = new_seq
    return preds
```

---

### `core/reporter.py`

```python
from fpdf import FPDF
import matplotlib.pyplot as plt
import os

def generate_report(results, timestamp):
    pdf_path = f"outputs/reports/Forecast_Report_{timestamp}.pdf"

    for name, preds in results.items():
        plt.figure(figsize=(10,4))
        plt.plot(preds, label=f"{name} Prediction")
        plt.title(f"{name} Forecast")
        plt.legend()
        plt.savefig(f"outputs/forecasts/{name}_forecast_{timestamp}.png")
        plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Enerwise Forecast Report", 0, 1, "C")
    pdf.set_font("Arial", "", 12)

    for name in results.keys():
        pdf.cell(0, 10, f"{name} Model Forecast", 0, 1)
        img_path = f"outputs/forecasts/{name}_forecast_{timestamp}.png"
        pdf.image(img_path, x=10, w=180)

    pdf.output(pdf_path)
    print(f"Report saved to {pdf_path}")
```

---


