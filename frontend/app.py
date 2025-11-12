

# import streamlit as st
# import requests
# import pandas as pd
# import uuid

# BACKEND_URL = "http://127.0.0.1:8000/chat"
# HEALTH_URL = "http://127.0.0.1:8000/health"

# st.set_page_config(page_title="Olist Data Copilot", layout="wide")

# # ping backend (best-effort)
# try:
#     requests.get(HEALTH_URL, timeout=2)
# except Exception:
#     pass

# # ---------------------------------------------------------
# # INIT MULTI-CHAT STATE
# # ---------------------------------------------------------
# def make_new_chat(title: str = "New chat"):
#     return {
#         "id": str(uuid.uuid4()),
#         "title": title,
#         "turns": [
#             {
#                 "role": "assistant",
#                 "payload": {
#                     "type": "text",
#                     "content": "Hi, I’m your Olist Data Copilot. Ask about orders, categories, customers, payments, delivery delays, or even say 'what is order_status'.",
#                 },
#             }
#         ],
#     }

# if "chats" not in st.session_state:
#     # start with a single chat
#     st.session_state.chats = [make_new_chat("Chat 1")]

# if "current_chat_id" not in st.session_state:
#     st.session_state.current_chat_id = st.session_state.chats[0]["id"]

# # helper to get current chat
# def get_current_chat():
#     for chat in st.session_state.chats:
#         if chat["id"] == st.session_state.current_chat_id:
#             return chat
#     # fallback – should not happen
#     st.session_state.current_chat_id = st.session_state.chats[0]["id"]
#     return st.session_state.chats[0]

# current_chat = get_current_chat()

# # ---------------------------------------------------------
# # SIDEBAR: CHAT LIST + ACTIONS
# # ---------------------------------------------------------
# with st.sidebar:
#     st.markdown("### 💬 Chats")
#     # list all chats
#     for chat in st.session_state.chats:
#         if st.button(chat["title"], key=f"chat-btn-{chat['id']}"):
#             st.session_state.current_chat_id = chat["id"]
#             st.rerun()

#     # new chat button
#     if st.button("➕ New chat"):
#         new_idx = len(st.session_state.chats) + 1
#         new_chat = make_new_chat(f"Chat {new_idx}")
#         st.session_state.chats.append(new_chat)
#         st.session_state.current_chat_id = new_chat["id"]
#         st.rerun()

#     st.markdown("---")
#     st.markdown("### Examples")
#     st.markdown("- orders per month")
#     st.markdown("- top 10 selling product categories")
#     st.markdown("- average order value by customer_state")
#     st.markdown("- what is order_status")
#     st.markdown("- translate to Hindi: delivered")
#     st.markdown("- where is my order 123")

#     # clear only current chat
#     if st.button("🗑 Clear current chat"):
#         current_chat["turns"] = [
#             {
#                 "role": "assistant",
#                 "payload": {
#                     "type": "text",
#                     "content": "Chat cleared. Ask again about Olist.",
#                 },
#             }
#         ]
#         st.rerun()

# # ---------------------------------------------------------
# # MAIN TITLE
# # ---------------------------------------------------------
# st.title("🛒 Olist Data Copilot")

# # ---------------------------------------------------------
# # RENDER CURRENT CHAT
# # ---------------------------------------------------------
# for turn in current_chat["turns"]:
#     role = turn["role"]
#     payload = turn["payload"]
#     with st.chat_message(role):
#         t = payload.get("type")
#         if t == "text":
#             st.write(payload["content"])
#         elif t == "table":
#             if "explanation" in payload:
#                 st.info(payload["explanation"])
#             # nicer: hide SQL in expander
#             with st.expander("Show generated SQL"):
#                 st.code(payload.get("sql", ""), language="sql")
#             df = pd.DataFrame(payload.get("data", []))
#             if len(df) == 0:
#                 st.warning("Query ran but returned 0 rows.")
#             else:
#                 st.markdown(f"**Rows:** {len(df)} • **Columns:** {len(df.columns)}")
#                 st.dataframe(df, use_container_width=True, hide_index=True)
#         else:
#             st.error(payload.get("error", "Unknown error"))
#             if "sql" in payload:
#                 st.code(payload["sql"], language="sql")

# # ---------------------------------------------------------
# # USER INPUT
# # ---------------------------------------------------------
# user_msg = st.chat_input("Try: 'top 10 selling categories' or 'average order value by state' or 'what is review_score'")
# if user_msg:
#     # append user turn to current chat
#     current_chat["turns"].append({"role": "user", "payload": {"type": "text", "content": user_msg}})

#     # build history for backend – last 10 messages of THIS chat
#     history = [
#         {
#             "role": t["role"],
#             "content": t["payload"].get("content", "") if isinstance(t["payload"], dict) else "",
#         }
#         for t in current_chat["turns"][-10:]
#     ]

#     payload = {
#         "message": user_msg,
#         "history": history,
#         "session_id": current_chat["id"],  # so backend can key by chat if it wants
#     }

#     try:
#         r = requests.post(BACKEND_URL, json=payload, timeout=12)
#         r.raise_for_status()
#         resp = r.json()
#     except Exception as e:
#         resp = {"type": "error", "error": f"Cannot reach backend: {e}"}

#     # append assistant turn to current chat
#     current_chat["turns"].append({"role": "assistant", "payload": resp})

#     # write back into session_state
#     for i, chat in enumerate(st.session_state.chats):
#         if chat["id"] == current_chat["id"]:
#             st.session_state.chats[i] = current_chat
#             break

#     st.rerun()


import streamlit as st
import requests
import pandas as pd
import uuid

BACKEND_URL = "http://127.0.0.1:8000/chat"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(page_title="Olist Data Copilot", layout="wide", page_icon="🛒")

# ---------- ping backend ----------
backend_ok = False
try:
    r = requests.get(HEALTH_URL, timeout=2)
    backend_ok = r.status_code == 200
except Exception:
    backend_ok = False

# ---------- global styles ----------
st.markdown(
    """
    <style>
    .main {
        background: #0f172a0d;
    }
    .stChatMessage {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }
    .assistant-bubble {
        background: #0f172a;
        color: white;
        padding: 0.9rem 1rem;
        border-radius: 1rem;
    }
    .user-bubble {
        background: #e2e8f0;
        color: #0f172a;
        padding: 0.9rem 1rem;
        border-radius: 1rem;
    }
    .streamlit-expanderHeader {
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- multi-chat state ----------
def make_new_chat(title: str = "New chat"):
    return {
        "id": str(uuid.uuid4()),
        "title": title,
        "turns": [
            {
                "role": "assistant",
                "payload": {
                    "type": "text",
                    "content": (
                        "Hi, I’m your Olist Data Copilot. Ask about orders, categories, customers, "
                        "payments, delivery delays, or even say 'what is order_status'."
                    ),
                },
            }
        ],
    }

if "chats" not in st.session_state:
    st.session_state.chats = [make_new_chat("Chat 1")]

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = st.session_state.chats[0]["id"]

def get_current_chat():
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.current_chat_id:
            return chat
    st.session_state.current_chat_id = st.session_state.chats[0]["id"]
    return st.session_state.chats[0]

current_chat = get_current_chat()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🛒 Olist Copilot")
    if backend_ok:
        st.success("Backend: online", icon="✅")
    else:
        st.error("Backend: offline", icon="⚠️")

    st.markdown("### Chats")
    for chat in st.session_state.chats:
        if st.button(chat["title"], key=f"chat-{chat['id']}"):
            st.session_state.current_chat_id = chat["id"]
            st.rerun()

    if st.button("➕ New chat"):
        new_idx = len(st.session_state.chats) + 1
        new_chat = make_new_chat(f"Chat {new_idx}")
        st.session_state.chats.append(new_chat)
        st.session_state.current_chat_id = new_chat["id"]
        st.rerun()

    if st.button("🗑 Clear current chat"):
        current_chat["turns"] = [
            {
                "role": "assistant",
                "payload": {"type": "text", "content": "Chat cleared. Ask again about Olist."},
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("### Example queries")
    st.markdown("- orders per month")
    st.markdown("- top 10 selling product categories")
    st.markdown("- average order value by customer_state")
    st.markdown("- what is order_status")
    st.markdown("- translate to Portuguese: delivered")
    st.markdown("- where is my order 47770eb9100c2d0c44946d9cf07ec65d")
    st.markdown("---")
    st.caption("Ask about orders, categories, payments, delivery, reviews…")

# ---------- TOP BAR ----------
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("Olist Data Copilot")
with col2:
    st.write("")
    st.caption("multi-chat • SQL • translations • tracking")

st.write("")

# ---------- RENDER CHAT ----------
for turn in current_chat["turns"]:
    role = turn["role"]
    payload = turn["payload"]
    t = payload.get("type")

    with st.chat_message(role):
        if t == "text":
            if role == "assistant":
                st.markdown(f"<div class='assistant-bubble'>{payload['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='user-bubble'>{payload['content']}</div>", unsafe_allow_html=True)

        elif t == "table":
            if "explanation" in payload:
                st.info(payload["explanation"])
            with st.expander("Show generated SQL"):
                st.code(payload.get("sql", ""), language="sql")
            df = pd.DataFrame(payload.get("data", []))
            if len(df) == 0:
                st.warning("Query ran but returned 0 rows.")
            else:
                st.markdown(f"**Rows:** {len(df)} • **Columns:** {len(df.columns)}")
                st.dataframe(df, use_container_width=True, hide_index=True)

        else:
            st.error(payload.get("error", "Unknown error from backend"))
            if "sql" in payload:
                st.code(payload["sql"], language="sql")

# ---------- USER INPUT ----------
user_msg = st.chat_input("Ask about orders, revenue, categories, delivery, translations...")
if user_msg:
    current_chat["turns"].append({"role": "user", "payload": {"type": "text", "content": user_msg}})

    history = [
        {
            "role": t["role"],
            "content": t["payload"].get("content", "") if isinstance(t["payload"], dict) else "",
        }
        for t in current_chat["turns"][-10:]
    ]
    payload = {
        "message": user_msg,
        "history": history,
        "session_id": current_chat["id"],
    }

    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=12)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        resp = {"type": "error", "error": f"Cannot reach backend: {e}"}

    current_chat["turns"].append({"role": "assistant", "payload": resp})

    for i, chat in enumerate(st.session_state.chats):
        if chat["id"] == current_chat["id"]:
            st.session_state.chats[i] = current_chat
            break

    st.rerun()
