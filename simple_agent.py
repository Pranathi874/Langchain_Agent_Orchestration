#step 1 : imports
import requests
import os
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
load_dotenv()

# 1st tool (search tool)
search_tool = DuckDuckGoSearchRun()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# 2nd tool(weather tool)

@tool
def get_place_temperature(city: str) -> dict:
    """Get the current weather of a given city."""
    
    data = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"appid": os.environ["OPENWEATHER_API_KEY"], "q": city , "units": "metric"},
        timeout=10
    ).json()
    
    if data.get("cod") != 200:
        raise RuntimeError(data)
    
    return {
        "city": data["name"],
        "temp_c": data["main"]["temp"],
        "condition": data["weather"][0]["description"]
 }

@tool
def calculator(expression: str) -> str:
    """Instant calculator via wolfram Alpha"""
    url = "http://api.wolframalpha.com/v2/simple"  #correct for results
    
    params = {
        "appid": os.environ["WOLFRAM_ALPHA_APPID"],
        "i": expression   # "18 * 1.8 + 32"
    }
    
    response = requests.get(url, params=params, timeout=5)
    return response.text.strip() if response.ok else f"Error: {response.status_code}"


#step 2: pull the react prompt from langchain hub
prompt = hub.pull("hwchase17/react") #pulls the standard react agent prompt
#print(hub.pull("hwchase17/react"))

#step 3: create the react agent manually with the pulled prompt
agent = create_react_agent(llm=model,
    tools=[search_tool, get_place_temperature],
    prompt=prompt
)

#step 4: Wrap the agent with an AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_place_temperature],
    verbose=True
)

#step 5: invoke the agent executor with a query
response = agent_executor.invoke({'input':"What is the capital of Tamil Nadu. Find it's current weather condition "})
print(response)