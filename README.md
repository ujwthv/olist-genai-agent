# Olist Data Copilot

GenAI agentic chatbot for the Brazilian Olist e-commerce dataset. You can ask in plain English:

- “orders per month”
- “top 10 selling product categories”
- “average order value by customer_state”
- “which product category was the highest selling in the past 2 quarters”
- “what is order_status”
- “translate to portuguese: delivered”

The backend turns your question → SQL for DuckDB → runs it on `olist.duckdb` → returns data + SQL + explanation. It tries Gemini first, then OpenRouter as fallback.

---

## 1. Features

- Chat-style UI (Streamlit).
- Conversational memory (last few turns sent to backend).
- LLM-to-SQL over real Olist tables.
- Dual LLM support:
  - Google Gemini (via Google AI Studio) – primary
  - OpenRouter (LLM of your choice) – fallback
- Auto-repair of typical LLM SQL mistakes (wrong alias, unqualified CTE).
- Returns explanation + exact SQL.
- Extra intents: definition (“what is order_status”), mock tracking, translation hook.

---

## 2. Prereqs

- Docker **(recommended)**
- Your own Olist DuckDB file at `backend/data/olist.duckdb`
- Your own API keys:
  - `GEMINI_API_KEY`
  - optionally `OPENROUTER_API_KEY`

If you don’t want Docker, you need Python 3.12+ and `pip`.

---

## 3. Run with Docker (recommended)

```bash


# 3. build & run
docker compose up --build

---

## 4. Run locally for python 3.12.x version
# backend
cd backend
python -m venv venv
venv/Scripts/activate  # on Windows
pip install -r requirements.txt
##Run FastAPI backend
uvicorn main:app --reload --port 8000

# Run Streamlit frontend (new terminal)
cd frontend
streamlit run app.py
