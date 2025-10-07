FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar arquivos do projeto
COPY . /app/

# Instalar dependências Python
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Criar diretórios necessários
RUN mkdir -p data/consumo data/pv outputs

# Porta do FastAPI
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
