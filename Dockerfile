# ── Base image ───────────────────────────────────────────────────────
FROM python:3.11-slim
 
# ── Set working directory ─────────────────────────────────────────────
WORKDIR /app
 
# ── Install dependencies first (cached layer) ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# ── Copy only what the API needs ──────────────────────────────────────
COPY api/       ./api/
COPY src/       ./src/
COPY model/     ./model/
 
# ── Expose port ───────────────────────────────────────────────────────
EXPOSE 8000
 
# ── Run the API ───────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
 
