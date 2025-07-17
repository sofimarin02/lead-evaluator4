FROM python:3.10-slim

# No generar .pyc y evitar cache
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar e instalar dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Documenta que escuchamos en 8080
EXPOSE 8080

# Opción A: Flask builtin (lee PORT en app.py)
CMD ["python", "app.py"]
