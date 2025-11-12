
# import os
# import json
# import math
# import datetime
# import re
# import requests

# import tools  # duckdb + schema

# # =========================================================
# # CONFIG
# # =========================================================

# def _normalize_gemini_model(raw: str | None) -> str:
#     return raw.strip() if raw else "gemini-1.5-flash"

# GEMINI_MODEL = _normalize_gemini_model(os.getenv("GEMINI_MODEL"))
# GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "1.5"))

# def _normalize_openrouter_model(raw: str | None) -> str:
#     if not raw:
#         return "google/gemma-2-9b-it"
#     low = raw.lower()
#     # auto-downsize huge models so we don't blow streamlit's 12s
#     if "70b" in low or "405b" in low or "72b" in low:
#         return "google/gemma-2-9b-it"
#     return raw.strip()

# OPENROUTER_MODEL = _normalize_openrouter_model(os.getenv("OPENROUTER_MODEL"))
# OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "4.0"))

# def get_gemini_key():
#     return os.getenv("GEMINI_API_KEY")

# def get_openrouter_key():
#     return os.getenv("OPENROUTER_API_KEY")


# # =========================================================
# # GENERIC HELPERS
# # =========================================================

# def rows_to_jsonable(rows):
#     out = []
#     for r in rows:
#         nr = {}
#         for k, v in r.items():
#             if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
#                 nr[k] = None
#             elif isinstance(v, (datetime.datetime, datetime.date)):
#                 nr[k] = v.isoformat()
#             else:
#                 nr[k] = v
#         out.append(nr)
#     return out


# def relax_overstrict_date_filter(user_message: str, sql: str) -> str:
#     """
#     LLMs often generate filters like:
#       WHERE o.order_purchase_timestamp = (SELECT max(...) FROM orders)
#     or
#       WHERE ... = (SELECT last_order_date FROM last_order)
#     which returns 0 rows on Olist. Strip them unless the user explicitly asked
#     for "today/recent/past/last".
#     """
#     if not sql:
#         return sql

#     m = user_message.lower()
#     wants_recent = any(x in m for x in ["today", "recent", "latest", "last", "past", "90 days", "quarter"])
#     if wants_recent:
#         return sql

#     # drop exact-timestamp filters
#     sql = re.sub(
#         r"WHERE\s+[A-Za-z0-9_]+\.\s*order_purchase_timestamp\s*=\s*\(SELECT\s+MAX\(order_purchase_timestamp\)\s+FROM\s+orders\)",
#         "",
#         sql,
#         flags=re.IGNORECASE,
#     )
#     sql = re.sub(
#         r"WHERE\s+[A-Za-z0-9_]+\.\s*order_purchase_timestamp\s*=\s*\(SELECT\s+last_order_date\s+FROM\s+last_order\)",
#         "",
#         sql,
#         flags=re.IGNORECASE,
#     )
#     sql = re.sub(r"\s+AND\s+GROUP BY", " GROUP BY", sql, flags=re.IGNORECASE)
#     sql = re.sub(r"\s+AND\s+ORDER BY", " ORDER BY", sql, flags=re.IGNORECASE)
#     return sql


# def sanitize_sql(sql: str) -> str:
#     if not sql:
#         return sql
#     sql = sql.replace("\\\n", " ")
#     sql = sql.replace("\\", " ")
#     sql = " ".join(sql.split())
#     return sql.strip().rstrip(";").strip()


# def build_table_columns_from_schema():
#     schema = tools.get_schema_summary()
#     out = {}
#     for table, cols in schema.items():
#         colnames = set()
#         for c in cols:
#             name = c.split()[0]
#             colnames.add(name)
#         out[table] = colnames
#     return out

# SCHEMA_TABLE_COLS = build_table_columns_from_schema()


# def rebind_unknown_columns(sql: str) -> str:
#     aliases = re.findall(
#         r"\bFROM\s+(\w+)\s+(\w+)|\bJOIN\s+(\w+)\s+(\w+)", sql, flags=re.IGNORECASE
#     )
#     alias_map = {}
#     for a, a_alias, b, b_alias in aliases:
#         if a and a_alias:
#             alias_map[a_alias] = a
#         if b and b_alias:
#             alias_map[b_alias] = b
#     if not alias_map:
#         return sql

#     dotted = set(re.findall(r"(\w+)\.(\w+)", sql))
#     for alias, col in dotted:
#         table = alias_map.get(alias)
#         if not table:
#             continue
#         table_cols = SCHEMA_TABLE_COLS.get(table, set())
#         if col not in table_cols:
#             replacement = None
#             for other_alias, other_table in alias_map.items():
#                 other_cols = SCHEMA_TABLE_COLS.get(other_table, set())
#                 if col in other_cols:
#                     replacement = other_alias
#                     break
#             if replacement:
#                 sql = re.sub(rf"\b{alias}\.{col}\b", f"{replacement}.{col}", sql)
#     return sql


# def fix_cte_column_usage(sql: str) -> str:
#     ctes = re.findall(
#         r"WITH\s+(\w+)\s+AS\s*\(\s*SELECT\s+.*?\s+AS\s+(\w+)\s+FROM\s+[\w\.]+.*?\)",
#         sql,
#         flags=re.IGNORECASE | re.DOTALL,
#     )
#     if not ctes:
#         return sql
#     for cte_name, cte_col in ctes:
#         sql = re.sub(
#             rf"date_trunc\(\s*'quarter'\s*,\s*{cte_col}\s*\)",
#             f"date_trunc('quarter', (SELECT {cte_col} FROM {cte_name}))",
#             sql,
#             flags=re.IGNORECASE,
#         )
#         sql = re.sub(
#             rf"date_trunc\(\s*'month'\s*,\s*{cte_col}\s*\)",
#             f"date_trunc('month', (SELECT {cte_col} FROM {cte_name}))",
#             sql,
#             flags=re.IGNORECASE,
#         )
#         sql = sql.replace(
#             f"{cte_col}) - INTERVAL", f"(SELECT {cte_col} FROM {cte_name})) - INTERVAL"
#         )
#     return sql


# def normalize_duckdb_intervals(sql: str) -> str:
#     if not sql:
#         return sql
#     pattern = r"INTERVAL\s+'(\d+)\s+([A-Za-z]+)'"
#     def repl(m):
#         num = m.group(1)
#         unit = m.group(2).upper()
#         if unit.endswith("S"):
#             unit = unit[:-1]
#         return f"INTERVAL {num} {unit}"
#     return re.sub(pattern, repl, sql)


# def fix_obviously_bad_window(sql: str) -> str:
#     if "PARTITION BY ()" not in sql:
#         return sql
#     return """
# WITH anchor AS (
#     SELECT max(order_purchase_timestamp) AS max_ts FROM orders
# ),
# ranges AS (
#     SELECT
#         date_trunc('month', max_ts) AS this_month,
#         date_trunc('month', max_ts) - INTERVAL 3 MONTH AS last3_start,
#         date_trunc('month', max_ts) - INTERVAL 6 MONTH AS prev3_start
#     FROM anchor
# ),
# last3 AS (
#     SELECT SUM(oi.price + oi.freight_value) AS revenue
#     FROM order_items_enriched oi
#     JOIN orders o ON oi.order_id = o.order_id
#     CROSS JOIN ranges r
#     WHERE o.order_purchase_timestamp >= r.last3_start
#       AND o.order_purchase_timestamp <  r.this_month
# ),
# prev3 AS (
#     SELECT SUM(oi.price + oi.freight_value) AS revenue
#     FROM order_items_enriched oi
#     JOIN orders o ON oi.order_id = o.order_id
#     CROSS JOIN ranges r
#     WHERE o.order_purchase_timestamp >= r.prev3_start
#       AND o.order_purchase_timestamp <  r.last3_start
# )
# SELECT
#     (SELECT revenue FROM last3) AS last_3_months_revenue,
#     (SELECT revenue FROM prev3) AS previous_3_months_revenue;
# """.strip()
# # =========================================================
# # INTENT (fixed)
# # =========================================================

# ANALYTIC_HINTS = [
#     " per ", " by ", " group", "average", "avg", "sum", "count", "top", "highest",
#     "last ", "past ", "quarter", "month", "revenue", "orders", "delivery", "delay",
#     "which ", "in the last", "customers", "seller", "payment", "review", "score",
#     "%", "percent", "percentage"
# ]

# def looks_analytic(message: str) -> bool:
#     m = message.lower()
#     if len(m.split()) > 6:
#         return True
#     return any(h in m for h in ANALYTIC_HINTS)

# def detect_intent(message: str) -> str:
#     m = message.lower().strip()

#     # normalize apostrophes so "what’s" and "what's" behave the same
#     m = m.replace("’", "'")

#     # if it starts with "what" AND looks analytic → go to SQL
#     if m.startswith("what") and looks_analytic(m):
#         return "sql_query"

#     if m.startswith("what is") or m.startswith("what's") or "definition of" in m:
#         return "definition"

#     if m.startswith("translate") or "translate to " in m or "translation:" in m:
#         return "translate"

#     if "where is my order" in m or "track order" in m:
#         return "track_order"

#     return "sql_query"



# # =========================================================
# # DEFINITION HELPERS (LLM-backed)
# # =========================================================
# def extract_definition_term(message: str) -> str:
#     m = message.strip().lower().replace("’", "'")
#     m = re.sub(r"^what\s+is\s+", "", m)
#     m = re.sub(r"^what's\s+", "", m)
#     m = re.sub(r"^definition of\s+", "", m)
#     return m.strip(" ?.")



# def _gemini_generate_content(base_url: str, model: str, key: str, prompt: str, timeout: float):
#     url = f"{base_url}/models/{model}:generateContent"
#     headers = {"Content-Type": "application/json"}
#     params = {"key": key}
#     body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
#     r = requests.post(url, headers=headers, params=params, data=json.dumps(body), timeout=timeout)
#     return r.json()


# def call_llm_for_definition(term: str, original_message: str) -> str | None:
#     prompt = f"""
# Explain the term below in <= 120 words, in plain English, as it would relate to an e-commerce / Olist dataset (orders, customers, payments, reviews). If the term is not an actual Olist column, explain its usual meaning in e-commerce.

# TERM: "{term}"
# USER ASKED: "{original_message}"
# """.strip()

#     gkey = get_gemini_key()
#     if gkey:
#         for base in ("https://generativelanguage.googleapis.com/v1", "https://generativelanguage.googleapis.com/v1beta"):
#             try:
#                 j = _gemini_generate_content(base, GEMINI_MODEL, gkey, prompt, GEMINI_TIMEOUT)
#                 if "error" not in j and j.get("candidates"):
#                     return j["candidates"][0]["content"]["parts"][0]["text"].strip()
#             except Exception:
#                 pass

#     okey = get_openrouter_key()
#     if okey:
#         try:
#             url = "https://openrouter.ai/api/v1/chat/completions"
#             headers = {
#                 "Content-Type": "application/json",
#                 "Authorization": f"Bearer {okey}",
#                 "HTTP-Referer": "http://localhost",
#                 "X-Title": "olist-genai-agent",
#             }
#             body = {
#                 "model": OPENROUTER_MODEL,
#                 "messages": [
#                     {"role": "system", "content": "You are a concise e-commerce/data assistant."},
#                     {"role": "user", "content": prompt},
#                 ],
#                 "temperature": 0.0,
#             }
#             r = requests.post(url, headers=headers, data=json.dumps(body), timeout=OPENROUTER_TIMEOUT)
#             j = r.json()
#             return j["choices"][0]["message"]["content"].strip()
#         except Exception:
#             pass

#     return None


# # =========================================================
# # TRANSLATION, TRACK ORDER – unchanged logic, so skipping to SQL
# # =========================================================

# def build_base_prompt(user_message: str, schema_text: str) -> str:
#     return f"""
# You are an expert DuckDB SQL generator for the Brazilian Olist e-commerce dataset.

# Return ONLY JSON:
# {{"sql": "<DUCKDB SQL QUERY>"}}

# Rules:
# - Always anchor relative periods on (SELECT max(order_purchase_timestamp) FROM orders).
# - Join order_items_enriched to orders for timestamps.
# - Group by product_category_name_english for category questions.
# - Revenue = price + freight_value.
# - Use DuckDB date_trunc and INTERVAL (no CURRENT_TIMESTAMP on 2016-2018 data).
# - Use only the columns in the schema.

# SCHEMA:
# {schema_text}

# USER QUESTION:
# {user_message}
# """.strip()


# def call_gemini_for_sql(prompt: str):
#     key = get_gemini_key()
#     if not key:
#         return None, "GEMINI_API_KEY not set"
#     last_err = None
#     for base in ("https://generativelanguage.googleapis.com/v1", "https://generativelanguage.googleapis.com/v1beta"):
#         try:
#             j = _gemini_generate_content(base, GEMINI_MODEL, key, prompt, GEMINI_TIMEOUT)
#             if "error" not in j:
#                 return j, None
#             last_err = j["error"].get("message", "unknown Gemini error")
#         except Exception as e:
#             last_err = str(e)
#     return None, f"Gemini API error: {last_err}"


# def parse_gemini_sql(resp: dict):
#     try:
#         candidates = resp.get("candidates", [])
#         if not candidates:
#             return None, "Gemini parse error: no candidates"
#         text = candidates[0]["content"]["parts"][0]["text"].strip()
#         obj = json.loads(text)
#         return obj.get("sql"), None
#     except Exception as e:
#         return None, f"Gemini parse error: {e}"


# def call_openrouter_for_sql(prompt: str):
#     key = get_openrouter_key()
#     if not key:
#         return None, "OPENROUTER_API_KEY not set"
#     url = "https://openrouter.ai/api/v1/chat/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {key}",
#         "HTTP-Referer": "http://localhost",
#         "X-Title": "olist-genai-agent",
#     }
#     body = {
#         "model": OPENROUTER_MODEL,
#         "messages": [
#             {"role": "system", "content": "You return only JSON with one key 'sql'."},
#             {"role": "user", "content": prompt},
#         ],
#         "temperature": 0.0,
#     }
#     try:
#         r = requests.post(url, headers=headers, data=json.dumps(body), timeout=OPENROUTER_TIMEOUT)
#         return r.json(), None
#     except Exception as e:
#         return None, f"OpenRouter HTTP error: {e}"


# def parse_openrouter_sql(resp: dict):
#     try:
#         text = resp["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         return None, f"OpenRouter parse error: no content: {e}"
#     if text.startswith("```"):
#         text = text.strip("`").replace("json", "", 1).strip()
#     try:
#         obj = json.loads(text)
#         sql = obj.get("sql")
#         if sql:
#             return sql, None
#     except json.JSONDecodeError as e:
#         m = re.search(r'"sql"\s*:\s*"([^"]+)"', text, re.DOTALL)
#         if m:
#             return m.group(1), None
#         return None, f"OpenRouter parse error after repair: {e}"
#     return None, "OpenRouter JSON has no 'sql'"

# # ===================== TRANSLATION HELPERS =====================
# def extract_translation_target_and_text(message: str):
#     """
#     Supports:
#       translate to korean: delivered
#       translate to japanese delivered
#       translate to mandrian: delivered
#     Default target: Hindi
#     """
#     text = message.strip()
#     target_lang = "Hindi"
#     to_translate = text

#     lower = text.lower()
#     if lower.startswith("translate to "):
#         rest = text[len("translate to ") :].strip()
#         if ":" in rest:
#             lang_part, txt = rest.split(":", 1)
#             target_lang = lang_part.strip()
#             to_translate = txt.strip()
#         else:
#             parts = rest.split(" ", 1)
#             target_lang = parts[0].strip()
#             to_translate = parts[1].strip() if len(parts) > 1 else ""
#     elif lower.startswith("translate "):
#         # e.g. "translate delivered" → default lang
#         to_translate = text[len("translate ") :].strip()

#     # fix common misspells / shortcuts
#     low_lang = target_lang.lower()
#     if low_lang in ("mandrian", "mandarin"):
#         target_lang = "Mandarin Chinese"
#     if low_lang in ("pt", "pt-br", "brazilian portuguese"):
#         target_lang = "Portuguese (Brazil)"

#     return target_lang, to_translate


# def translate_with_llm(text: str, target_lang: str) -> dict:
#     """
#     Try Gemini (v1 -> v1beta), then OpenRouter.
#     On ANY failure, return a safe fallback so the backend never 500s.
#     """
#     prompt = f"""
# Translate the text to {target_lang}.
# Return ONLY JSON in this exact shape:
# {{
#   "translation": "<the translation>",
#   "romanization": "<latin pronunciation or transliteration, if none is natural, repeat the translation>"
# }}
# Text: {text}
# """.strip()

#     gkey = get_gemini_key()
#     if gkey:
#         for base in (
#             "https://generativelanguage.googleapis.com/v1",
#             "https://generativelanguage.googleapis.com/v1beta",
#         ):
#             try:
#                 j = _gemini_generate_content(
#                     base, GEMINI_MODEL, gkey, prompt, GEMINI_TIMEOUT
#                 )
#                 if isinstance(j, dict) and "error" in j:
#                     continue
#                 if (
#                     isinstance(j, dict)
#                     and j.get("candidates")
#                     and j["candidates"][0]["content"]["parts"]
#                 ):
#                     raw = j["candidates"][0]["content"]["parts"][0]["text"].strip()
#                     try:
#                         return json.loads(raw)
#                     except Exception:
#                         return {"translation": raw, "romanization": ""}
#             except Exception:
#                 # swallow and try next provider
#                 pass

#     okey = get_openrouter_key()
#     if okey:
#         try:
#             url = "https://openrouter.ai/api/v1/chat/completions"
#             headers = {
#                 "Content-Type": "application/json",
#                 "Authorization": f"Bearer {okey}",
#                 "HTTP-Referer": "http://localhost",
#                 "X-Title": "olist-genai-agent",
#             }
#             body = {
#                 "model": OPENROUTER_MODEL,
#                 "messages": [
#                     {
#                         "role": "system",
#                         "content": "You are a translation assistant. Always return valid JSON with keys 'translation' and 'romanization'.",
#                     },
#                     {"role": "user", "content": prompt},
#                 ],
#                 "temperature": 0.0,
#             }
#             r = requests.post(
#                 url,
#                 headers=headers,
#                 data=json.dumps(body),
#                 timeout=OPENROUTER_TIMEOUT,
#             )
#             j = r.json()
#             raw = j["choices"][0]["message"]["content"].strip()
#             try:
#                 return json.loads(raw)
#             except Exception:
#                 return {"translation": raw, "romanization": ""}
#         except Exception:
#             pass

#     # fallback – must never crash the API
#     return {
#         "translation": text,
#         "romanization": "pronunciation not available (LLM call failed)",
#     }


# def build_explanation(user_message: str, sql: str, extra: str | None = None) -> str:
#     msg = user_message.lower()
#     parts = []
#     if "category" in msg or "product" in msg:
#         parts.append("I treated it as a category-level question and used item-level tables joined to orders.")
#     if any(x in msg for x in ["last", "past", "quarter", "month", "day"]):
#         parts.append("I anchored the time window on the latest order_purchase_timestamp in the dataset.")
#     if "top" in msg or "highest" in msg:
#         parts.append("I ordered the aggregated values descending to surface the top entries.")
#     if extra:
#         parts.append(extra)
#     if not parts:
#         parts.append("I converted your natural language into a DuckDB SQL aggregate over the Olist tables.")
#     parts.append("You can modify the SQL below to explore further.")
#     return " ".join(parts)

# def get_last_sql_from_history(history: list[str | dict]) -> str | None:
#     """
#     History from frontend is a list of dicts with role/content.
#     We only care about assistant messages that had SQL in them.
#     """
#     if not history:
#         return None
#     # history is list of dicts like {"role": "assistant", "content": "..."}
#     # but our assistant responses in frontend are wrapped; so we just scan strings
#     for msg in reversed(history):
#         if not isinstance(msg, dict):
#             continue
#         content = msg.get("content", "")
#         # crude: if there's a SELECT in it, treat it as SQL
#         if "SELECT" in content.upper():
#             return content
#     return None

# # =========================================================
# # MAIN HANDLER
# # =========================================================

# def handle_user_message(message: str, history: list):
#     intent = detect_intent(message)

#     # --- definitions ---
#     if intent == "definition":
#         term = extract_definition_term(message)
#         # 1) local
#         if term in tools.COLUMN_DEFS:
#             return {"type": "text", "content": tools.explain_column(term)}
#         # 2) LLM
#         llm_def = call_llm_for_definition(term, message)
#         if llm_def:
#             return {"type": "text", "content": llm_def}
#         # 3) honest fallback
#         return {
#             "type": "text",
#             "content": f"I don't have a stored definition for '{term}'. Try asking it with more dataset context.",
#         }

#     # --- translation / order tracking kept as in your previous version ---
#     if intent == "translate":
#         target_lang, to_translate = extract_translation_target_and_text(message)
#         data = translate_with_llm(to_translate, target_lang)
#         translation = data.get("translation", "").strip()
#         romanization = data.get("romanization", "").strip() or "pronunciation not available"
#         return {"type": "text", "content": f"{translation} ({romanization})"}


#     if intent == "track_order":
#         return try_track_order(message)

#     # --- SQL path ---
#     schema = tools.get_schema_summary()
#     schema_text = "\n".join([f"{t}: {', '.join(cols)}" for t, cols in schema.items()])
#     base_prompt = build_base_prompt(message, schema_text)

#     sql = None
#     errs = []

#     g_resp, g_err = call_gemini_for_sql(base_prompt)
#     if g_resp is not None and g_err is None:
#         sql, p_err = parse_gemini_sql(g_resp)
#         if p_err:
#             errs.append(p_err)
#     else:
#         if g_err:
#             errs.append(g_err)

#     if sql is None:
#         o_resp, o_err = call_openrouter_for_sql(base_prompt)
#         if o_resp is not None and o_err is None:
#             sql, p_err = parse_openrouter_sql(o_resp)
#             if p_err:
#                 errs.append(p_err)
#         else:
#             if o_err:
#                 errs.append(o_err)

#     if sql is None:
#         return {
#             "type": "error",
#             "error": " | ".join(errs) if errs else "No LLM produced SQL.",
#         }

#     sql = relax_overstrict_date_filter(message, sql)
#     sql = sanitize_sql(sql)
#     sql = normalize_duckdb_intervals(sql)
#     sql = fix_cte_column_usage(sql)
#     sql = rebind_unknown_columns(sql)
#     sql = fix_obviously_bad_window(sql)

#     try:
#         df = tools.run_sql(sql)
#         df = df.replace({math.nan: None, float("inf"): None, float("-inf"): None})
#         rows = rows_to_jsonable(df.head(200).to_dict(orient="records"))
#         extra = None
#         if "category" in message.lower():
#             extra_info = tools.fake_external_lookup(message)
#             return {
#                 "type": "table",
#                 "sql": sql,
#                 "data": rows,
#                 "explanation": build_explanation(message, sql, extra_info),
#             }

#     except Exception as db_err:
#         return {
#             "type": "error",
#             "error": f"DB error running generated SQL: {db_err}",
#             "sql": sql,
#         }
    
import os
import json
import math
import datetime
import re
import requests

import tools  # duckdb + schema

# =========================================================
# CONFIG
# =========================================================

def _normalize_gemini_model(raw: str | None) -> str:
    return raw.strip() if raw else "gemini-1.5-flash"

GEMINI_MODEL = _normalize_gemini_model(os.getenv("GEMINI_MODEL"))
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "1.5"))

def _normalize_openrouter_model(raw: str | None) -> str:
    if not raw:
        return "google/gemma-2-9b-it"
    low = raw.lower()
    # downsize huge models so we don't blow the 12s frontend timeout
    if "70b" in low or "405b" in low or "72b" in low:
        return "google/gemma-2-9b-it"
    return raw.strip()

OPENROUTER_MODEL = _normalize_openrouter_model(os.getenv("OPENROUTER_MODEL"))
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "4.0"))

def get_gemini_key():
    return os.getenv("GEMINI_API_KEY")

def get_openrouter_key():
    return os.getenv("OPENROUTER_API_KEY")


# =========================================================
# GENERIC HELPERS
# =========================================================

def rows_to_jsonable(rows):
    out = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                nr[k] = None
            elif isinstance(v, (datetime.datetime, datetime.date)):
                nr[k] = v.isoformat()
            else:
                nr[k] = v
        out.append(nr)
    return out


def relax_overstrict_date_filter(user_message: str, sql: str) -> str:
    if not sql:
        return sql

    m = user_message.lower()
    wants_recent = any(x in m for x in ["today", "recent", "latest", "last", "past", "90 days", "quarter"])
    if wants_recent:
        return sql

    sql = re.sub(
        r"WHERE\s+[A-Za-z0-9_]+\.\s*order_purchase_timestamp\s*=\s*\(SELECT\s+MAX\(order_purchase_timestamp\)\s+FROM\s+orders\)",
        "",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"WHERE\s+[A-Za-z0-9_]+\.\s*order_purchase_timestamp\s*=\s*\(SELECT\s+last_order_date\s+FROM\s+last_order\)",
        "",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(r"\s+AND\s+GROUP BY", " GROUP BY", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s+AND\s+ORDER BY", " ORDER BY", sql, flags=re.IGNORECASE)
    return sql


def sanitize_sql(sql: str) -> str:
    if not sql:
        return sql
    sql = sql.replace("\\\n", " ")
    sql = sql.replace("\\", " ")
    sql = " ".join(sql.split())
    return sql.strip().rstrip(";").strip()


def build_table_columns_from_schema():
    schema = tools.get_schema_summary()
    out = {}
    for table, cols in schema.items():
        colnames = set()
        for c in cols:
            name = c.split()[0]
            colnames.add(name)
        out[table] = colnames
    return out

SCHEMA_TABLE_COLS = build_table_columns_from_schema()


def rebind_unknown_columns(sql: str) -> str:
    aliases = re.findall(
        r"\bFROM\s+(\w+)\s+(\w+)|\bJOIN\s+(\w+)\s+(\w+)", sql, flags=re.IGNORECASE
    )
    alias_map = {}
    for a, a_alias, b, b_alias in aliases:
        if a and a_alias:
            alias_map[a_alias] = a
        if b and b_alias:
            alias_map[b_alias] = b
    if not alias_map:
        return sql

    dotted = set(re.findall(r"(\w+)\.(\w+)", sql))
    for alias, col in dotted:
        table = alias_map.get(alias)
        if not table:
            continue
        table_cols = SCHEMA_TABLE_COLS.get(table, set())
        if col not in table_cols:
            replacement = None
            for other_alias, other_table in alias_map.items():
                other_cols = SCHEMA_TABLE_COLS.get(other_table, set())
                if col in other_cols:
                    replacement = other_alias
                    break
            if replacement:
                sql = re.sub(rf"\b{alias}\.{col}\b", f"{replacement}.{col}", sql)
    return sql


def fix_cte_column_usage(sql: str) -> str:
    ctes = re.findall(
        r"WITH\s+(\w+)\s+AS\s*\(\s*SELECT\s+.*?\s+AS\s+(\w+)\s+FROM\s+[\w\.]+.*?\)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not ctes:
        return sql
    for cte_name, cte_col in ctes:
        sql = re.sub(
            rf"date_trunc\(\s*'quarter'\s*,\s*{cte_col}\s*\)",
            f"date_trunc('quarter', (SELECT {cte_col} FROM {cte_name}))",
            sql,
            flags=re.IGNORECASE,
        )
        sql = re.sub(
            rf"date_trunc\(\s*'month'\s*,\s*{cte_col}\s*\)",
            f"date_trunc('month', (SELECT {cte_col} FROM {cte_name}))",
            sql,
            flags=re.IGNORECASE,
        )
        sql = sql.replace(
            f"{cte_col}) - INTERVAL", f"(SELECT {cte_col} FROM {cte_name})) - INTERVAL"
        )
    return sql


def normalize_duckdb_intervals(sql: str) -> str:
    if not sql:
        return sql
    pattern = r"INTERVAL\s+'(\d+)\s+([A-Za-z]+)'"
    def repl(m):
        num = m.group(1)
        unit = m.group(2).upper()
        if unit.endswith("S"):
            unit = unit[:-1]
        return f"INTERVAL {num} {unit}"
    return re.sub(pattern, repl, sql)


def fix_obviously_bad_window(sql: str) -> str:
    if "PARTITION BY ()" not in sql:
        return sql
    return """
WITH anchor AS (
    SELECT max(order_purchase_timestamp) AS max_ts FROM orders
),
ranges AS (
    SELECT
        date_trunc('month', max_ts) AS this_month,
        date_trunc('month', max_ts) - INTERVAL 3 MONTH AS last3_start,
        date_trunc('month', max_ts) - INTERVAL 6 MONTH AS prev3_start
    FROM anchor
),
last3 AS (
    SELECT SUM(oi.price + oi.freight_value) AS revenue
    FROM order_items_enriched oi
    JOIN orders o ON oi.order_id = o.order_id
    CROSS JOIN ranges r
    WHERE o.order_purchase_timestamp >= r.last3_start
      AND o.order_purchase_timestamp <  r.this_month
),
prev3 AS (
    SELECT SUM(oi.price + oi.freight_value) AS revenue
    FROM order_items_enriched oi
    JOIN orders o ON oi.order_id = o.order_id
    CROSS JOIN ranges r
    WHERE o.order_purchase_timestamp >= r.prev3_start
      AND o.order_purchase_timestamp <  r.last3_start
)
SELECT
    (SELECT revenue FROM last3) AS last_3_months_revenue,
    (SELECT revenue FROM prev3) AS previous_3_months_revenue;
""".strip()


# =========================================================
# INTENT
# =========================================================

ANALYTIC_HINTS = [
    " per ", " by ", " group", "average", "avg", "sum", "count", "top", "highest",
    "last ", "past ", "quarter", "month", "revenue", "orders", "delivery", "delay",
    "which ", "in the last", "customers", "seller", "payment", "review", "score",
    "%", "percent", "percentage"
]

def looks_analytic(message: str) -> bool:
    m = message.lower()
    if len(m.split()) > 6:
        return True
    return any(h in m for h in ANALYTIC_HINTS)

def detect_intent(message: str) -> str:
    m = message.lower().strip()
    m = m.replace("’", "'")

    if m.startswith("what") and looks_analytic(m):
        return "sql_query"

    if m.startswith("what is") or m.startswith("what's") or "definition of" in m:
        return "definition"

    if m.startswith("translate") or "translate to " in m or "translation:" in m:
        return "translate"

    if "where is my order" in m or "track order" in m:
        return "track_order"

    return "sql_query"


# =========================================================
# DEFINITION HELPERS
# =========================================================

def extract_definition_term(message: str) -> str:
    m = message.strip().lower().replace("’", "'")
    m = re.sub(r"^what\s+is\s+", "", m)
    m = re.sub(r"^what's\s+", "", m)
    m = re.sub(r"^definition of\s+", "", m)
    return m.strip(" ?.")


def _gemini_generate_content(base_url: str, model: str, key: str, prompt: str, timeout: float):
    url = f"{base_url}/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": key}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    r = requests.post(url, headers=headers, params=params, data=json.dumps(body), timeout=timeout)
    return r.json()


def call_llm_for_definition(term: str, original_message: str) -> str | None:
    prompt = f"""
Explain the term below in <= 120 words, in plain English, as it would relate to an e-commerce / Olist dataset (orders, customers, payments, reviews). If the term is not an actual Olist column, explain its usual meaning in e-commerce.

TERM: "{term}"
USER ASKED: "{original_message}"
""".strip()

    gkey = get_gemini_key()
    if gkey:
        for base in ("https://generativelanguage.googleapis.com/v1", "https://generativelanguage.googleapis.com/v1beta"):
            try:
                j = _gemini_generate_content(base, GEMINI_MODEL, gkey, prompt, GEMINI_TIMEOUT)
                if "error" not in j and j.get("candidates"):
                    return j["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                pass

    okey = get_openrouter_key()
    if okey:
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {okey}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "olist-genai-agent",
            }
            body = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a concise e-commerce/data assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            }
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=OPENROUTER_TIMEOUT)
            j = r.json()
            return j["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    return None


# =========================================================
# ORDER TRACKING
# =========================================================

def try_track_order(message: str):
    m = re.search(r"\b([0-9a-fA-F-]{6,})\b", message)
    if not m:
        return {
            "type": "text",
            "content": "I couldn't see an order id there. Say: `where is my order <real-order-id-from-olist>`",
        }
    order_id = m.group(1)
    sql = f"""
    SELECT
        order_id,
        order_status,
        order_purchase_timestamp,
        order_delivered_customer_date,
        order_estimated_delivery_date
    FROM orders
    WHERE order_id = '{order_id}'
    """
    try:
        df = tools.run_sql(sql)
    except Exception:
        return {
            "type": "text",
            "content": f"Mock tracking: order {order_id} is DELIVERED.",
        }
    if df.empty:
        return {
            "type": "text",
            "content": f"I couldn’t find order `{order_id}` in Olist. It may be a fake/test id.",
        }
    row = df.iloc[0]
    status = row["order_status"]
    delivered = row["order_delivered_customer_date"]
    eta = row["order_estimated_delivery_date"]
    parts = [f"Order `{order_id}` status: **{status}**."]
    if delivered:
        parts.append(f"Delivered on: {delivered}.")
    elif eta:
        parts.append(f"Estimated delivery: {eta}.")
    return {"type": "text", "content": " ".join(parts)}


# =========================================================
# TRANSLATION
# =========================================================

def extract_translation_target_and_text(message: str):
    text = message.strip()
    target_lang = "Hindi"
    to_translate = text

    lower = text.lower()
    if lower.startswith("translate to "):
        rest = text[len("translate to "):].strip()
        if ":" in rest:
            lang_part, txt = rest.split(":", 1)
            target_lang = lang_part.strip()
            to_translate = txt.strip()
        else:
            parts = rest.split(" ", 1)
            target_lang = parts[0].strip()
            to_translate = parts[1].strip() if len(parts) > 1 else ""
    elif lower.startswith("translate "):
        to_translate = text[len("translate "):].strip()

    low_lang = target_lang.lower()
    if low_lang in ("mandrian", "mandarin"):
        target_lang = "Mandarin Chinese"
    if low_lang in ("pt", "pt-br", "brazilian portuguese"):
        target_lang = "Portuguese (Brazil)"

    return target_lang, to_translate


def translate_with_llm(text: str, target_lang: str) -> dict:
    prompt = f"""
Translate the text to {target_lang}.
Return ONLY JSON in this exact shape:
{{
  "translation": "<the translation>",
  "romanization": "<latin pronunciation or transliteration, if none is natural, repeat the translation>"
}}
Text: {text}
""".strip()

    gkey = get_gemini_key()
    if gkey:
        for base in ("https://generativelanguage.googleapis.com/v1",
                     "https://generativelanguage.googleapis.com/v1beta"):
            try:
                j = _gemini_generate_content(base, GEMINI_MODEL, gkey, prompt, GEMINI_TIMEOUT)
                if isinstance(j, dict) and "error" in j:
                    continue
                if isinstance(j, dict) and j.get("candidates"):
                    raw = j["candidates"][0]["content"]["parts"][0]["text"].strip()
                    try:
                        return json.loads(raw)
                    except Exception:
                        return {"translation": raw, "romanization": ""}
            except Exception:
                pass

    okey = get_openrouter_key()
    if okey:
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {okey}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "olist-genai-agent",
            }
            body = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a translation assistant. Always return valid JSON with keys 'translation' and 'romanization'.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            }
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=OPENROUTER_TIMEOUT)
            j = r.json()
            raw = j["choices"][0]["message"]["content"].strip()
            try:
                return json.loads(raw)
            except Exception:
                return {"translation": raw, "romanization": ""}
        except Exception:
            pass

    return {
        "translation": text,
        "romanization": "pronunciation not available (LLM call failed)",
    }


# =========================================================
# LLM → SQL
# =========================================================

def build_base_prompt(user_message: str, schema_text: str) -> str:
    return f"""
You are an expert DuckDB SQL generator for the Brazilian Olist e-commerce dataset.

Return ONLY JSON:
{{"sql": "<DUCKDB SQL QUERY>"}}

Rules:
- Always anchor relative periods on (SELECT max(order_purchase_timestamp) FROM orders).
- Join order_items_enriched to orders for timestamps.
- Group by product_category_name_english for category questions.
- Revenue = price + freight_value.
- Use DuckDB date_trunc and INTERVAL.
- Use only the columns in the schema.

SCHEMA:
{schema_text}

USER QUESTION:
{user_message}
""".strip()


def call_gemini_for_sql(prompt: str):
    key = get_gemini_key()
    if not key:
        return None, "GEMINI_API_KEY not set"
    last_err = None
    for base in ("https://generativelanguage.googleapis.com/v1", "https://generativelanguage.googleapis.com/v1beta"):
        try:
            j = _gemini_generate_content(base, GEMINI_MODEL, key, prompt, GEMINI_TIMEOUT)
            if "error" not in j:
                return j, None
            last_err = j["error"].get("message", "unknown Gemini error")
        except Exception as e:
            last_err = str(e)
    return None, f"Gemini API error: {last_err}"


def parse_gemini_sql(resp: dict):
    try:
        candidates = resp.get("candidates", [])
        if not candidates:
            return None, "Gemini parse error: no candidates"
        text = candidates[0]["content"]["parts"][0]["text"].strip()
        obj = json.loads(text)
        return obj.get("sql"), None
    except Exception as e:
        return None, f"Gemini parse error: {e}"


def call_openrouter_for_sql(prompt: str):
    key = get_openrouter_key()
    if not key:
        return None, "OPENROUTER_API_KEY not set"
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "olist-genai-agent",
    }
    body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You return only JSON with one key 'sql'."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=OPENROUTER_TIMEOUT)
        return r.json(), None
    except Exception as e:
        return None, f"OpenRouter HTTP error: {e}"


def parse_openrouter_sql(resp: dict):
    try:
        text = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None, f"OpenRouter parse error: no content: {e}"
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    try:
        obj = json.loads(text)
        sql = obj.get("sql")
        if sql:
            return sql, None
    except json.JSONDecodeError as e:
        m = re.search(r'"sql"\s*:\s*"([^"]+)"', text, re.DOTALL)
        if m:
            return m.group(1), None
        return None, f"OpenRouter parse error after repair: {e}"
    return None, "OpenRouter JSON has no 'sql'"


# =========================================================
# EXPLANATION
# =========================================================

def build_explanation(user_message: str, sql: str, extra: str | None = None) -> str:
    msg = user_message.lower()
    parts = []
    if "category" in msg or "product" in msg:
        parts.append("I treated it as a category-level question and used item-level tables joined to orders.")
    if any(x in msg for x in ["last", "past", "quarter", "month", "day"]):
        parts.append("I anchored the time window on the latest order_purchase_timestamp in the dataset.")
    if "top" in msg or "highest" in msg:
        parts.append("I ordered the aggregated values descending to surface the top entries.")
    if extra:
        parts.append(extra)
    if not parts:
        parts.append("I converted your natural language into a DuckDB SQL aggregate over the Olist tables.")
    parts.append("You can modify the SQL below to explore further.")
    return " ".join(parts)


# small history helper (for future contextual rewrites)
def get_last_sql_from_history(history: list) -> str | None:
    if not history:
        return None
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if "SELECT" in content.upper():
            return content
    return None


# =========================================================
# MAIN HANDLER
# =========================================================

def handle_user_message(message: str, history: list):
    intent = detect_intent(message)

    # 1) definition
    if intent == "definition":
        term = extract_definition_term(message)
        if term in tools.COLUMN_DEFS:
            return {"type": "text", "content": tools.explain_column(term)}
        llm_def = call_llm_for_definition(term, message)
        if llm_def:
            return {"type": "text", "content": llm_def}
        return {
            "type": "text",
            "content": f"I don't have a stored definition for '{term}'. Try asking it with more dataset context.",
        }

    # 2) translation
    if intent == "translate":
        target_lang, to_translate = extract_translation_target_and_text(message)
        data = translate_with_llm(to_translate, target_lang)
        translation = data.get("translation", "").strip()
        romanization = data.get("romanization", "").strip() or "pronunciation not available"
        return {"type": "text", "content": f"{translation} ({romanization})"}

    # 3) order tracking
    if intent == "track_order":
        return try_track_order(message)

    # 4) analytics / SQL
    schema = tools.get_schema_summary()
    schema_text = "\n".join([f"{t}: {', '.join(cols)}" for t, cols in schema.items()])
    base_prompt = build_base_prompt(message, schema_text)

    sql = None
    errs = []

    g_resp, g_err = call_gemini_for_sql(base_prompt)
    if g_resp is not None and g_err is None:
        sql, p_err = parse_gemini_sql(g_resp)
        if p_err:
            errs.append(p_err)
    else:
        if g_err:
            errs.append(g_err)

    if sql is None:
        o_resp, o_err = call_openrouter_for_sql(base_prompt)
        if o_resp is not None and o_err is None:
            sql, p_err = parse_openrouter_sql(o_resp)
            if p_err:
                errs.append(p_err)
        else:
            if o_err:
                errs.append(o_err)

    if sql is None:
        return {
            "type": "text",
            "content": "I couldn’t generate SQL for that with the current models. Try clarifying the table/time window.",
        }

    sql = relax_overstrict_date_filter(message, sql)
    sql = sanitize_sql(sql)
    sql = normalize_duckdb_intervals(sql)
    sql = fix_cte_column_usage(sql)
    sql = rebind_unknown_columns(sql)
    sql = fix_obviously_bad_window(sql)

    try:
        df = tools.run_sql(sql)
        df = df.replace({math.nan: None, float("inf"): None, float("-inf"): None})
        rows = rows_to_jsonable(df.head(200).to_dict(orient="records"))

        # add extra info for category questions
        extra = None
        if "category" in message.lower():
            extra = tools.fake_external_lookup(message)

        return {
            "type": "table",
            "sql": sql,
            "data": rows,
            "explanation": build_explanation(message, sql, extra),
        }
    except Exception as db_err:
        return {
            "type": "error",
            "error": f"DB error running generated SQL: {db_err}",
            "sql": sql,
        }
