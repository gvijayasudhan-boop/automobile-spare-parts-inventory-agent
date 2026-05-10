import streamlit as st
import boto3
import json
import os
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────
REGION          = "ap-southeast-1"
LAMBDA_FUNCTION = "AutoPartsInventoryAgent_Ankush_Singapore"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Auto Parts Inventory Agent",
    page_icon  = "🚗",
    layout     = "wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A1628 0%, #065A82 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: #00D4AA; margin: 0; font-size: 1.6rem; }
    .main-header p  { color: #8FB3CC; margin: 0.3rem 0 0; font-size: 0.9rem; }

    .chat-msg-user {
        background: #0D2137;
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #E0EEF8;
        text-align: right;
    }
    .chat-msg-bot {
        background: #0A1628;
        border: 1px solid #065A82;
        border-radius: 4px 12px 12px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #BDD9EF;
    }
    .redirect-msg {
        background: #1A1205;
        border: 1px solid #F5A623;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #F5CBA7;
    }
    .error-msg {
        background: #1A0505;
        border: 1px solid #E74C3C;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #F1948A;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚗 Automobile Spare Parts Inventory Agent</h1>
    <p>ML + GenAI Powered  |  Amazon SageMaker · DynamoDB · Amazon Bedrock · RAG</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Quick Actions")
    if st.button("🔴 Critical Parts",     use_container_width=True):
        st.session_state.quick_query = "Which parts are critically low right now?"
    if st.button("🚙 Honda City Parts",   use_container_width=True):
        st.session_state.quick_query = "Which parts are low for Honda City?"
    if st.button("🚗 Maruti Swift Parts", use_container_width=True):
        st.session_state.quick_query = "Which spare parts are needed for Maruti Swift 2021?"
    if st.button("📦 Restock Engine Oil", use_container_width=True):
        st.session_state.quick_query = "How much engine oil 5W-30 should we reorder and from which supplier?"
    if st.button("📋 Return Policy",      use_container_width=True):
        st.session_state.quick_query = "What is the return policy for overstocked OEM parts?"
    if st.button("🌧️ Monsoon Prep",       use_container_width=True):
        st.session_state.quick_query = "What parts should we stock up for the monsoon season?"

    st.markdown("---")
    st.markdown("### ℹ️ Architecture")
    st.markdown("""
- **ML**: SageMaker XGBoost
- **DB**: DynamoDB (composite key)
- **AI**: Amazon Bedrock Nova Lite
- **RAG**: S3 Vectors
- **Logic**: AWS Lambda
""")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quick_query" not in st.session_state:
    st.session_state.quick_query = ""

# ── Lambda caller ──────────────────────────────────────────────────────────
def call_lambda(query: str) -> dict:
    try:
        lam = boto3.client("lambda", region_name=REGION)
        response = lam.invoke(
            FunctionName   = LAMBDA_FUNCTION,
            InvocationType = "RequestResponse",
            Payload        = json.dumps({"query": query}).encode(),
        )
        result = json.loads(response["Payload"].read())
        if "body" in result:
            body = json.loads(result["body"]) if isinstance(result["body"], str) else result["body"]
            return body
        return result
    except Exception as e:
        return {
            "response": f"Could not reach Lambda: {str(e)}. Please check your AWS credentials and region.",
            "type": "error"
        }

# ── Display chat history ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-msg-user">💬 {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        msg_type  = msg.get("type", "info")
        css_class = "redirect-msg" if msg_type == "redirect" else \
                    "error-msg"    if msg_type == "error"    else "chat-msg-bot"
        icon      = "↩️" if msg_type == "redirect" else \
                    "⚠️" if msg_type == "error"    else "🤖"
        st.markdown(
            f'<div class="{css_class}">{icon} {msg["content"]}</div>',
            unsafe_allow_html=True
        )

# ── Chat input ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    default_val = st.session_state.get("quick_query", "")
    user_input  = st.text_input(
        "Ask about parts, stock levels, or compatibility...",
        value            = default_val,
        placeholder      = "e.g. Which brake pads are low for Honda City 2022?",
        label_visibility = "collapsed",
        key              = "chat_input",
    )
with col2:
    send_clicked = st.button("Send 🚀", use_container_width=True, type="primary")

# Clear quick query after use
if st.session_state.quick_query:
    st.session_state.quick_query = ""

# ── Process query ──────────────────────────────────────────────────────────
if send_clicked and user_input.strip():
    query = user_input.strip()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Call Lambda
    with st.spinner("Checking inventory + forecasts..."):
        result = call_lambda(query)

    response_text = result.get("response", "Sorry, I could not process that request.")
    response_type = result.get("type", "info")

    # Add bot response
    st.session_state.messages.append({
        "role":    "assistant",
        "content": response_text,
        "type":    response_type,
    })
    st.rerun()

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<p style='text-align:center; color:#4A6FA5; font-size:0.8rem;'>"
    f"ML + GenAI Automobile Spare Parts Inventory Agent  |  POC Demo  |  "
    f"{datetime.now().strftime('%d %b %Y')}</p>",
    unsafe_allow_html=True
)