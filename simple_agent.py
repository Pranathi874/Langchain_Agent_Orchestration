import os
from dotenv import load_dotenv

load_dotenv()   

from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.tools import DuckDuckGoSearchRun, tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub

# =========================
# LLM
# =========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    api_key=GOOGLE_API_KEY,   # 🔥 EXPLICIT PASS
    streaming=False
)

# =========================
# TOOLS
# =========================
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [search_tool, calculator]

# =========================
# PROMPTS
# =========================
SUMMARY_PROMPT = PromptTemplate.from_template("""
Summarize the following content in 100–150 words.

Content:
{text}
""")

EMAIL_PROMPT = PromptTemplate.from_template("""
Write a professional and polite email based on the summary.

Summary:
{text}
""")

summary_chain = SUMMARY_PROMPT | llm | StrOutputParser()
email_chain = EMAIL_PROMPT | llm | StrOutputParser()

# =========================
# RESEARCH AGENT (ReAct)
# =========================
react_prompt = hub.pull("hwchase17/react")

research_agent = AgentExecutor(
    agent=create_react_agent(llm, tools, react_prompt),
    tools=tools,
    handle_parsing_errors=True,
    verbose=False
)

# =========================
# ORCHESTRATOR
# =========================
def orchestrator(topic: str) -> Dict[str, Any]:
    research = research_agent.invoke(
        {"input": f"Research the topic: {topic}"}
    )["output"]

    summary = summary_chain.invoke({"text": research})
    email = email_chain.invoke({"text": summary})

    return {
        "research": research,
        "summary": summary,
        "email": email
    }
