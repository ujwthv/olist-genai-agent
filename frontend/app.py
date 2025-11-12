import streamlit as st
import requests
import pandas as pd

BACKEND_URL = "http://127.0.0.1:8000/chat"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(page_title="Olist Data Copilot", layout="wide")

# ping backend
try:
    requests.get(HEALTH_URL, timeout=2)
except Exception:
    pass

if "turns" not in st.session_state:
    st.session_state.turns = [
        {"role": "assistant", "payload": {"type": "text", "content": "Hi, I’m your Olist Data Copilot. Ask about orders, categories, customers, payments, delivery delays, or even say 'what is order_status'."}}
    ]

st.title("🛒 Olist Data Copilot")

for turn in st.session_state.turns:
    role = turn["role"]
    payload = turn["payload"]
    with st.chat_message(role):
        t = payload.get("type")
        if t == "text":
            st.write(payload["content"])
        elif t == "table":
            if "explanation" in payload:
                st.info(payload["explanation"])
            st.code(payload.get("sql", ""), language="sql")
            df = pd.DataFrame(payload.get("data", []))
            if len(df) == 0:
                st.warning("Query ran but returned 0 rows.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error(payload.get("error", "Unknown error"))
            if "sql" in payload:
                st.code(payload["sql"], language="sql")

user_msg = st.chat_input("Try: 'top 10 selling categories' or 'average order value by state' or 'what is review_score'")
if user_msg:
    st.session_state.turns.append({"role": "user", "payload": {"type": "text", "content": user_msg}})
    payload = {
        "message": user_msg,
        "history": [
            {
                "role": t["role"],
                "content": t["payload"].get("content", "") if isinstance(t["payload"], dict) else "",
            }
            for t in st.session_state.turns[-10:]
        ]
    }
    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=12)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        resp = {"type": "error", "error": f"Cannot reach backend: {e}"}
    st.session_state.turns.append({"role": "assistant", "payload": resp})
    st.rerun()

with st.sidebar:
    st.markdown("### Examples")
    st.markdown("- orders per month")
    st.markdown("- top 10 selling product categories")
    st.markdown("- average order value by customer_state")
    st.markdown("- what is order_status")
    st.markdown("- translate to Hindi: delivered")
    st.markdown("- where is my order 123")
    if st.button("Clear chat"):
        st.session_state.turns = [
            {"role": "assistant", "payload": {"type": "text", "content": "Chat cleared. Ask again about Olist."}}
        ]
        st.rerun()
