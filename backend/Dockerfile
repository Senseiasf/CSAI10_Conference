# ==========================================
# Stage 1: Build dependencies
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ==========================================
# Stage 2: Production runtime (non-root)
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Создаем непривилегированного пользователя appuser (UID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser

# Копируем и устанавливаем подготовленные wheels
COPY --from=builder /build/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt && \
    rm -rf /wheels requirements.txt

# Копируем исходный код приложения
COPY app /app/app

# Назначаем права доступа пользователю appuser
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_PORT=8080 \
    APP_HOST=0.0.0.0

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
