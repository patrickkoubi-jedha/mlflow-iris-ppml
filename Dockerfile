FROM python:3.11-slim

WORKDIR /app

# Installation des outils système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY . .

# Création d'un utilisateur non-root (bonne pratique)
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

CMD ["python", "train_mlflow35_iris.py"]
