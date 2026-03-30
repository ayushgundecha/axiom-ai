# axiom-ai server — Python + Playwright + Chromium
# Multi-stage build, non-root user, health check

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY axiom/ ./axiom/

RUN pip install --no-cache-dir .

# Install Playwright browsers
RUN playwright install chromium && playwright install-deps chromium


FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

# Copy application code
COPY axiom/ ./axiom/
COPY tasks/ ./tasks/

# Create trajectories directory
RUN mkdir -p /app/trajectories

# Create non-root user
RUN useradd --create-home appuser
# Playwright needs writable cache — copy to appuser
RUN cp -r /root/.cache /home/appuser/.cache && chown -R appuser:appuser /home/appuser/.cache
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "axiom.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
