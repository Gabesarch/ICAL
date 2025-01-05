# main.py

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from real_time_agent_runner import RealTimeAgentRunner

app = FastAPI()

runner: Optional[RealTimeAgentRunner] = None

class StartRequest(BaseModel):
    config_file: str
    user_intent: str
    human_feedback_enabled: bool = False
    model: str = "gpt4o"

class FeedbackRequest(BaseModel):
    feedback: str

@app.post("/start")
def start_agent(req: StartRequest):
    global runner
    runner = RealTimeAgentRunner(
        human_feedback_enabled=req.human_feedback_enabled,
        model=req.model
    )
    runner.setup(req.config_file, req.user_intent)
    return {
        "status": "Agent started",
        "human_feedback_enabled": req.human_feedback_enabled,
        "model": req.model,
        "screenshot_base64": runner.get_screenshot_b64()
    }

@app.get("/propose")
def propose_action():
    """
    1. If feedback is enabled, propose action only (no step).
    2. If feedback is disabled, step the environment automatically.
    """
    global runner
    if not runner:
        return {"action": "No runner available", "screenshot_base64": ""}
    action_str, screenshot_b64 = runner.propose_action()
    return {
        "action": action_str,
        "screenshot_base64": screenshot_b64
    }

@app.post("/feedback")
def feedback_agent(req: FeedbackRequest):
    """
    Apply human feedback to revise the proposed action, if feedback is enabled.
    """
    global runner
    if not runner:
        return {"action": "No runner available"}
    revised_str = runner.apply_feedback(req.feedback)
    return {"action": revised_str}

@app.get("/commit")
def commit_action():
    """
    Commit the proposed action to the environment if feedback is enabled.
    """
    global runner
    if not runner:
        return {"action": "No runner available", "screenshot_base64": ""}
    action_str, screenshot_b64 = runner.commit_action()
    return {
        "action": action_str,
        "screenshot_base64": screenshot_b64
    }

@app.get("/stop")
def stop_agent():
    """
    Stop the environment session.
    """
    global runner
    if not runner:
        return {"status": "No runner to stop"}
    status_msg = runner.stop()
    runner = None
    return {"status": status_msg}

# Serve static front-end
from fastapi.staticfiles import StaticFiles
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
def read_index():
    return {"message": "Go to /public/index.html to use the UI."}

if __name__ == "__main__":
    # run server
    uvicorn.run(app, host="127.0.0.1", port=8000)
