# Enerwise AI Forecast Service

Serviço FastAPI para previsão de consumo energético usando pipeline Super-Híbrido.

##  Funcionalidades

- Previsão com múltiplos modelos (LSTM, Transformer, XGBoost, Prophet)
- API REST com FastAPI
- Dockerizado com suporte a GPU
- Pronto para deployment em cloud

##  Instalação

### Local
``
pip install -r requirements.txt
python main.py
Com Docker
bash
docker build -t enerwise-ai .
docker run -p 8000:8000 enerwise-ai
 API Endpoints
Health Check
bash
GET /health
Previsão
bash
POST /forecast/
{
  "horizon": 1,
  "mode": "light"
}
Estrutura
main.py - Servidor FastAPI

pipeline.py - Pipeline Super-Híbrido

models.py - Modelos de ML

utils.py - Funções auxiliares

config.py - Configurações

Dockerfile - Containerização
