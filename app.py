import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
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
# TOOLS (ADDED — REQUIRED FOR AGENT)
# =========================

# 1️⃣ DuckDuckGo Web Search
search_tool = DuckDuckGoSearchRun()

# 2️⃣ Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=4000,
    load_all_available_meta=True
)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# 3️⃣ ArXiv Tool
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000
)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Toolkit (LLM + Tools = Agent concept)
tools = [
    Tool(
        func=search_tool.run,
        name="web_search",
        description="Search the web for current events and real-time data."
    ),
    Tool(
        func=wiki_tool.run,
        name="wikipedia",
        description="Search Wikipedia for historical and general facts."
    ),
    Tool(
        func=arxiv_tool.run,
        name="arxiv_research",
        description=(
            "Search for scholarly articles and research papers. Covers "
            "Physics, Mathematics, Computer Science, Quantitative Biology, "
            "Quantitative Finance, Statistics, Electrical Engineering, and Economics."
        )
    )
]

# =========================
# PROMPTS (UNCHANGED)
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
# CHAINS (UNCHANGED)
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
    seen_urls = set()

    queries = [
        query,
        f"{query} overview",
        f"{query} research",
        f"{query} analysis",
        f"{query} latest trends"
    ]

    with DDGS() as ddgs:
        for q in queries:
            try:
                results = ddgs.text(q, max_results=5)
                for r in results:
                    url = r.get("href")
                    if url and url not in seen_urls:
                        sources.append({
                            "title": r.get("title", "Source"),
                            "url": url
                        })
                        seen_urls.add(url)

                    if len(sources) >= 10:
                        return sources
            except Exception:
                continue

    return sources


# =========================
# ORCHESTRATOR (UNCHANGED OUTPUT)
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