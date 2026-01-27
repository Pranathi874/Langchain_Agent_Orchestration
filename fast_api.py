import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from simple_agent import orchestrator

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Multi-Agent Research API (Groq + LLaMA)",
    version="1.0.0"
)

# =========================
# REQUEST MODEL
# =========================
class TopicInput(BaseModel):
    topic: str

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "status": "API running",
        "engine": "Groq + LLaMA",
        "agents": ["Researcher", "Critic", "Fact Checker", "Writer"]
    }

# =========================
# MAIN ENDPOINT
# =========================
@app.post("/run")
def run_agents(data: TopicInput):

    if not data.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    try:
        result = orchestrator(data.topic)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )

    # Save output for audit / debugging
    try:
        with open("agent_output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    except Exception:
        pass  # Non-critical

    return result
