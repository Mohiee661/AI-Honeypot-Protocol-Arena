from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from honeypot_env import HoneypotEnv

app = FastAPI(title="Honeypot Arena", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = HoneypotEnv()


class ResetRequest(BaseModel):
    seed: int = 0
    agent_id: str = "untrusted_agent"


class StepRequest(BaseModel):
    action: str


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    return env.reset(seed=req.seed)


@app.post("/step")
def step(req: StepRequest):
    if env.episode_number < 0:
        raise HTTPException(status_code=400, detail="Call /reset before /step")
    return env.step(req.action)


@app.get("/state")
def state():
    return env.get_state()
