# Dockerfile
FROM python:3.12-slim

# system deps if duckdb needs them
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy backend + frontend
COPY backend /app/backend
COPY frontend /app/frontend
# COPY .env.example /app/.env.example

# install deps (put a combined requirements in backend/requirements.txt)
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

RUN pip install --no-cache-dir streamlit pandas

# expose backend (FastAPI) and frontend (Streamlit)
EXPOSE 8000
EXPOSE 8501

# tiny runner that starts both
CMD sh -c "cd /app/backend && uvicorn main:app --host 0.0.0.0 --port 8000 & \
           cd /app/frontend && streamlit run app.py --server.address 0.0.0.0 --server.port 8501 && \
           wait"
