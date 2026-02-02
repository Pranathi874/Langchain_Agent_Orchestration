# =========================
# SILENCE WARNINGS & LOGS
# =========================
import os
import warnings
import logging
warnings.filterwarnings("ignore")

logging.getLogger().setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# =========================
# IMPORTS
# =========================
import streamlit as st
from dotenv import load_dotenv
from app import orchestrator
from datetime import datetime
import time
import base64

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="logo.png",
    layout="wide"
)

# =========================
# GLOBAL CSS (IMPORTANT)
# =========================
st.markdown("""
<style>

/* Global text */
html, body, [class*="st-"] {
    color: #ffffff !important;
}

/* Background */
.stApp {
    background:
        radial-gradient(circle at top left, rgba(114,9,183,0.35), transparent 45%),
        radial-gradient(circle at bottom right, rgba(226,189,107,0.28), transparent 50%),
        linear-gradient(135deg, #4D067B, #2b033f);
}

/* Labels */
label {
    color: #e0d6ff !important;
}

/* TextArea */
textarea {
    background: linear-gradient(135deg, #1a1b3a, #2b2d5c) !important;
    color: #ffffff !important;
    border: 1px solid #00ddff !important;
    border-radius: 14px !important;
}

textarea::placeholder {
    color: #b8b8ff !important;
}

textarea:focus {
    box-shadow: 0 0 14px rgba(0,221,255,0.8) !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #ffffff !important;
}

/* Alerts */
.stAlert {
    color: #ffffff !important;
}


/* Primary button */
div.stButton > button {
    background: linear-gradient(135deg, #6a00ff, #3b0a77) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.6em 1.4em !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35) !important;
    transition: all 0.25s ease-in-out !important;
}

/* Hover effect */
div.stButton > button:hover {
    background: linear-gradient(135deg, #8b2cff, #5a189a) !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(0,0,0,0.45) !important;
}

/* Click (active) */
div.stButton > button:active {
    transform: scale(0.98);
}



</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def load_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

top_logo_base64 = load_logo_base64("logo_.png")
title_logo_base64 = load_logo_base64("robo.png")

# =========================
# TOP HEADER
# =========================
st.markdown(f"""
<div style="
    background: rgba(255,255,255,0.12);
    padding: 14px 26px;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
">
    <div style="display:flex; align-items:center; gap:16px;">
        <img src="data:image/png;base64,{top_logo_base64}" style="height:50px;" />
        <div>
            <div style="font-size:16px;font-weight:700;color:#ffffff;">AGENT COLLAB</div>
            <div style="font-size:12px;color:#e0d6ff;">ORCHESTRATOR</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# MAIN TITLE
# =========================
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown(f"""
    <div style="display:flex;justify-content:center;align-items:center;gap:14px;">
        <img src="data:image/png;base64,{title_logo_base64}" style="height:42px;" />
        <h1 style="margin:0;color:#ffffff;font-weight:700;">
            Multi-Agent Research System
        </h1>
    </div>
    """, unsafe_allow_html=True)

now = datetime.now()
st.markdown(
    f"<p style='text-align:center;color:#cfcfff;'>🗒 {now.strftime('%A, %d %B %Y | %I:%M %p')}</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# API KEY CHECK
# =========================
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found in environment variables")
    st.stop()

# =========================
# INPUT SECTION
# =========================
st.markdown("<h3 style='color:#ffffff;'>🔎 Research Topic</h3>", unsafe_allow_html=True)
st.caption("Fast • Intelligent • Multi-Agent Research Engine")

topic = st.text_area(
    "✎ Enter a research topic",
    placeholder="E.g., Gold rates in India",
    height=110
)

# =========================
# TABS
# =========================
tabs = st.tabs([
    "🔍 Research",
    "🧠 Critic Review",
    "🔗 Sources",
    "✅ Fact Check",
    "📊 Insights",
    "📄 Summary",
    "✉️ Email",
    "🏷️ Titles"
])

research_ph = tabs[0].empty()
critic_ph = tabs[1].empty()
sources_ph = tabs[2].empty()
fact_ph = tabs[3].empty()
insights_ph = tabs[4].empty()
summary_ph = tabs[5].empty()
email_ph = tabs[6].empty()
titles_ph = tabs[7].empty()

research_ph.info("No research yet. Start a query to see results.")

# =========================
# ACTION
# =========================
if st.button("🚀 Start Research"):
    if not topic.strip():
        st.warning("⚠️ Please enter a topic")
    else:
        progress = st.progress(0)

        with st.spinner("🤝 Agents collaborating..."):
            try:
                progress.progress(30)
                output = orchestrator(topic)
                progress.progress(100)
            except Exception as e:
                st.error(str(e))
                st.stop()

        research_ph.subheader("Detailed Research")
        research_ph.write(output["research"])

        critic_ph.subheader("Critic Feedback")
        critic_ph.json(output["critic_feedback"])

        sources_ph.subheader("Sources")
        for s in output["sources"]:
            sources_ph.markdown(f"- [{s['title']}]({s['url']})")

        fact_ph.subheader("Fact Check")
        fact_ph.write(output["fact_check"])

        insights_ph.subheader("Key Insights")
        insights_ph.write(output["insights"])

        summary_ph.subheader("Summary")
        summary_ph.write(output["summary"])

        email_ph.subheader("Professional Email")
        email_ph.write(output["email"])

        titles_ph.subheader("Generated Titles")
        titles_ph.write(output["titles"])

# =========================
# FOOTER
# =========================
st.markdown(
    "<hr><p style='text-align:center;color:#cfcfff;'>© 2026 Multi-Agent Research System</p>",
    unsafe_allow_html=True
)
