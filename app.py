import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ddgs import DDGS

# =========================
# LOAD ENV
# =========================
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found")

# =========================
# LLM CONFIG
# =========================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=900
)

# =========================
# PROMPTS
# =========================
RESEARCH_PROMPT = PromptTemplate.from_template(
    """
    Write a detailed, well-structured research article on the topic below.
    Include explanation, applications, benefits, challenges, and examples.
    Do NOT mention searching or sources explicitly.

    Topic: {topic}
    """
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    "Summarize the following research clearly:\n\n{text}"
)

FACT_CHECK_PROMPT = PromptTemplate.from_template(
    "Fact-check the following research and mention inaccuracies if any:\n\n{text}"
)

INSIGHT_PROMPT = PromptTemplate.from_template(
    "Extract key insights and implications:\n\n{text}"
)

EMAIL_PROMPT = PromptTemplate.from_template(
    "Write a professional email based on this summary:\n\n{text}"
)

TITLE_PROMPT = PromptTemplate.from_template(
    "Generate 5 short, catchy titles:\n\n{text}"
)

CRITIC_PROMPT = PromptTemplate.from_template(
    """
Critically evaluate the following research and return ONLY valid JSON
(do not add explanations, markdown, or extra text).

Format EXACTLY like this:

{{
  "strengths": [],
  "weaknesses": [],
  "missing_points": [],
  "suggestions": []
}}

Research:
{text}
"""
)



# =========================
# CHAINS
# =========================
research_chain = RESEARCH_PROMPT | llm | StrOutputParser()
summary_chain = SUMMARY_PROMPT | llm | StrOutputParser()
fact_chain = FACT_CHECK_PROMPT | llm | StrOutputParser()
insight_chain = INSIGHT_PROMPT | llm | StrOutputParser()
email_chain = EMAIL_PROMPT | llm | StrOutputParser()
title_chain = TITLE_PROMPT | llm | StrOutputParser()
critic_chain = CRITIC_PROMPT | llm | StrOutputParser()

# =========================
# SOURCES (10 RESULTS)
# =========================
def collect_sources(query: str) -> List[Dict[str, str]]:
    sources = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=10):
            if r.get("href"):
                sources.append({
                    "title": r.get("title", "Source"),
                    "url": r["href"]
                })
    return sources

# =========================
# ORCHESTRATOR (FINAL)
# =========================
def orchestrator(topic: str) -> Dict[str, Any]:
    research = research_chain.invoke({"topic": topic})

    summary = summary_chain.invoke({"text": research})
    insights = insight_chain.invoke({"text": research})
    fact_check = fact_chain.invoke({"text": research})
    email = email_chain.invoke({"text": summary})
    titles = title_chain.invoke({"text": summary})
    critic_feedback = critic_chain.invoke({"text": research})

    return {
        "research": research,
        "critic_feedback": critic_feedback,
        "sources": collect_sources(topic),
        "fact_check": fact_check,
        "insights": insights,
        "summary": summary,
        "email": email,
        "titles": titles
    }
