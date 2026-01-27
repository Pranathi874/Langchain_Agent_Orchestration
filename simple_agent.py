import requests
import os
import numexpr
from dotenv import load_dotenv

from langchain_community.tools import tool, DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# LLM
# =========================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompt
prompt = hub.pull("hwchase17/react")

# =========================
# TOOLS
# =========================

# 1. Search tool
search_tool = DuckDuckGoSearchRun()

# 2. Weather tool
@tool
def get_weather_data(city: str) -> dict:
    """Fetch current weather data."""
    data = requests.get(
        "http://api.weatherstack.com/current",
        params={
            "access_key": os.getenv("WEATHERSTACK_API_KEY"),
            "query": city
        },
        timeout=10
    ).json()

    if "current" not in data:
        raise RuntimeError(data)

    return {
        "city": city,
        "temperature": data["current"]["temperature"],
        "condition": data["current"]["weather_descriptions"][0]
    }

# 3. Calculator tool
@tool
def calculator(expression: str) -> float:
    """Evaluate a math expression safely."""
    return float(numexpr.evaluate(expression))

# =========================
# AGENT
# =========================
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data, calculator],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data, calculator],
    verbose=True
)

# =========================
# RUN
# =========================
response = agent_executor.invoke({
    "input": "find the capital of India, then find its current temperature and subtract 5 from it"
})

print(response)
