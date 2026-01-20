import os
import streamlit as st
from dotenv import load_dotenv

from simple_agent import orchestrator

# =========================
# LOAD ENV
# =========================
load_dotenv()

st.set_page_config(page_title="LangChain Multi-Agent System")
st.title("🤖 LangChain Multi-Agent System")
st.write("Enter a topic below to start research")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found")
    st.stop()

# =========================
# UI
# =========================
topic = st.text_input(
    "Enter a topic",
    placeholder="gold rates in india"
)

if st.button("Start Research"):
    if not topic.strip():
        st.warning("Please enter a topic")
    else:
        with st.spinner("Researching..."):
            output = orchestrator(topic)

        tab1, tab2, tab3 = st.tabs(
            ["🔍 Research Data", "📄 Summary", "✉️ Email"]
        )

        with tab1:
            st.subheader("Raw Research Data")
            st.write(output["research"])

        with tab2:
            st.subheader("Summary (100–150 words)")
            st.write(output["summary"])

        with tab3:
            st.subheader("Email Content")
            st.write(output["email"])
