import duckdb
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve()
BACKEND_DIR = HERE.parent
APP_DIR = BACKEND_DIR.parent
ROOT_DIR = APP_DIR.parent

CANDIDATE_PATHS = [
    BACKEND_DIR / "data" / "olist.duckdb",
    APP_DIR / "data" / "olist.duckdb",
    ROOT_DIR / "data" / "olist.duckdb",
]

DB_PATH = None
for p in CANDIDATE_PATHS:
    if p.exists():
        DB_PATH = p
        break

if DB_PATH is not None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
else:
    con = None  # backend can still start

def run_sql(sql: str):
    if con is None:
        return pd.DataFrame([])
    return con.execute(sql).fetchdf()

def get_schema_summary():
    if con is None:
        return {}
    q = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema='main'
    ORDER BY table_name, ordinal_position
    """
    df = con.execute(q).fetchdf()
    tables = {}
    for _, row in df.iterrows():
        tables.setdefault(row["table_name"], []).append(
            f"{row['column_name']} {row['data_type']}"
        )
    return tables

CATEGORY_FACTS = {
    "electronics": (
        "Electronics in Olist often show up as 'eletronicos' or 'informatica_acessorios'. "
        "They have higher item value and relatively lower freight ratio."
    ),
    "furniture_decor": (
        "Furniture/decor is bulkier → higher freight_value and usually longer delivery time."
    ),
    "bed_bath_table": (
        "Household categories like bed_bath_table are frequent and often appear in smaller-ticket orders."
    ),
}

ECOM_GLOSSARY = {
    "aov": "Average order value = total revenue / number of orders.",
    "freight_value": "Shipping amount recorded for the order item.",
    "order_status": "order_status is the final state of the order (delivered, shipped, canceled, unavailable, etc.).",
    "churn": "Customer who purchased and then stopped buying in a later period.",
}

COLUMN_DEFS = {
    "order_status": "order_status is the final state of the order (delivered, shipped, canceled, unavailable, etc.).",
    "order_purchase_timestamp": "Timestamp when the customer placed the order.",
    "payment_value": "Amount recorded for that payment entry.",
    "review_score": "Customer satisfaction score (1–5).",
}

def explain_column(name: str) -> str:
    name = name.lower()
    if name in COLUMN_DEFS:
        return COLUMN_DEFS[name]
    if name in ECOM_GLOSSARY:
        return ECOM_GLOSSARY[name]
    return (
        f"I don’t have a stored description for `{name}`, but it looks like a column or term from the Olist dataset."
    )

def get_category_facts(name: str) -> str | None:
    if not name:
        return None
    key = name.lower().strip()
    if key in CATEGORY_FACTS:
        return CATEGORY_FACTS[key]
    for k, v in CATEGORY_FACTS.items():
        if k in key or key in k:
            return v
    return None

def fake_external_lookup(query: str) -> str:
    return (
        f"(stub) No extra external info found for: '{query}'. "
        "In a full build this would query a product/RAG index."
    )
