# =========================
# SILENCE ALL WARNINGS & LOGS
# =========================
import os
import warnings
import logging

warnings.filterwarnings("ignore")

# Silence logging from noisy libraries
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ddgs").setLevel(logging.ERROR)

# HuggingFace + tokenizers silence
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# =========================
# IMPORTS
# =========================
import streamlit as st
from dotenv import load_dotenv
from simple_agent import orchestrator

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🤖",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🤖 Multi-Agent Research System (Groq + LLaMA)")
st.caption("Fast • Intelligent • Multi-Agent Research Engine")

# =========================
# API KEY CHECK
# =========================
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found in environment variables")
    st.stop()

# =========================
# USER INPUT
# =========================
topic = st.text_input(
    "🔎 Enter a research topic",
    placeholder="Gold rates in India"
)



# =========================
# ACTION
# =========================
if st.button("🚀 Start Research"):
    if not topic.strip():
        st.warning("⚠️ Please enter a topic before starting")
    else:
        progress = st.progress(0)

        with st.spinner("🤝 Agents collaborating..."):
            try:
                progress.progress(30)
                output = orchestrator(topic)
                progress.progress(100)
            except Exception as e:
                st.error(f"❌ Research failed: {str(e)}")
                st.stop()

        # =========================
        # RESULTS TABS
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

        with tabs[0]:
            st.subheader("Detailed Research")
            st.write(output["research"])

        with tabs[1]:
            st.subheader("Critic Feedback")
            st.json(output["critic_feedback"])

        with tabs[2]:
            st.subheader("Sources")
            for s in output["sources"]:
                st.markdown(f"- [{s['title']}]({s['url']})")

        with tabs[3]:
            st.subheader("Fact Check")
            st.write(output["fact_check"])

        with tabs[4]:
            st.subheader("Key Insights")
            st.write(output["insights"])

        with tabs[5]:
            st.subheader("Summary")
            st.write(output["summary"])

        with tabs[6]:
            st.subheader("Professional Email")
            st.write(output["email"])

        with tabs[7]:
            st.subheader("Generated Titles")
            st.write(output["titles"])
