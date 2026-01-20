import json
from pydantic import BaseModel
from fastapi import FastAPI

from simple_agent import orchestrator

# =====================================
# FASTAPI APP
# =====================================
app = FastAPI(
    title="LangChain Multi-Agent System",
    description="Research → Summary → Email using LangChain Agents",
    version="1.0"
)

# =====================================
# INPUT MODEL
# =====================================
class TopicInput(BaseModel):
    topic: str

# =====================================
# ROOT ENDPOINT (FIXES 404 ISSUE)
# =====================================
@app.get("/")
def home():
    return {
        "message": "LangChain Multi-Agent API is running successfully",
        "use_endpoint": "/run",
        "method": "POST"
    }

# =====================================
# MAIN AGENT ENDPOINT
# =====================================
@app.post("/run")
def run_agents(data: TopicInput):
    result = orchestrator(data.topic)

    # Save output for automation
    with open("agent_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result
