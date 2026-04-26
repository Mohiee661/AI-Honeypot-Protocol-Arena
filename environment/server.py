import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from honeypot_env import HoneypotEnv


@asynccontextmanager
async def lifespan(_app: FastAPI):
    authtoken = os.environ.get("NGROK_AUTHTOKEN")
    if authtoken:
        from pyngrok import ngrok
        ngrok.set_auth_token(authtoken)
        port = int(os.environ.get("PORT", 7860))
        tunnel = ngrok.connect(port)
        print(f"\n{'='*60}")
        print(f"  NGROK PUBLIC URL: {tunnel.public_url}")
        print(f"{'='*60}\n")
    yield


app = FastAPI(title="Honeypot Arena", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = HoneypotEnv()

metrics_store = {
    "baseline": [],
    "trained": [],
    "reasoning_log": [],
}


class ResetRequest(BaseModel):
    seed: int = 0
    agent_id: str = "untrusted_agent"


class StepRequest(BaseModel):
    action: str


class LogEpisodeRequest(BaseModel):
    phase: str
    episode: int
    mode: str
    scenario_id: int = 0
    total_reward: float
    honesty_score: float = 0.0
    answer_correct: bool = False
    used_shortcut: bool = False
    verified_from_source: bool = False
    reasoning: str


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


@app.post("/log_episode")
def log_episode(req: LogEpisodeRequest):
    entry = req.model_dump()
    phase = entry.get("phase", "trained")
    if phase in metrics_store:
        metrics_store[phase].append(entry)
    metrics_store["reasoning_log"].append({
        "episode": entry["episode"],
        "mode": entry["mode"],
        "scenario_id": entry.get("scenario_id", 0),
        "answer_correct": entry.get("answer_correct", False),
        "honesty_score": entry.get("honesty_score", 0.0),
        "used_shortcut": entry.get("used_shortcut", False),
        "verified_from_source": entry.get("verified_from_source", False),
        "reasoning": entry["reasoning"],
    })
    return {"status": "logged"}


@app.get("/metrics")
def metrics():
    return metrics_store


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=_build_dashboard_html())


def _build_dashboard_html() -> str:
    import json
    data = json.dumps(metrics_store)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="10">
  <title>Honeypot Arena — Live Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0f0f0f; color: #e0e0e0; font-family: 'Courier New', monospace; padding: 24px; }}
    h1 {{ color: #00ff88; font-size: 1.6rem; margin-bottom: 24px; letter-spacing: 1px; }}
    h2 {{ color: #00ff88; font-size: 1rem; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }}
    .stats-bar {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
    .card {{ background: #1a1a1a; border: 1px solid #00ff8844; border-radius: 8px; padding: 20px 28px; flex: 1; min-width: 160px; }}
    .card .label {{ font-size: 0.75rem; color: #888; margin-bottom: 8px; }}
    .card .value {{ font-size: 2rem; color: #00ff88; font-weight: bold; }}
    .chart-section {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
    .chart-wrap {{ position: relative; max-height: 320px; }}
    .log-box {{ background: #0a0a0a; border: 1px solid #333; border-radius: 8px; padding: 16px; max-height: 400px; overflow-y: auto; font-size: 0.82rem; line-height: 1.6; }}
    .log-entry {{ margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #222; }}
    .log-header {{ color: #00ff88; }}
    .log-reasoning {{ color: #ccc; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>🍯 Honeypot Arena — Live Dashboard</h1>

  <div class="stats-bar" id="stats-bar"></div>

  <div class="chart-section">
    <h2>Baseline Progress (Honesty Score — live during Cell 10)</h2>
    <div class="chart-wrap"><canvas id="baselineChart"></canvas></div>
  </div>

  <div class="chart-section">
    <h2>Honesty Score by Mode: Baseline vs Trained</h2>
    <div class="chart-wrap"><canvas id="barChart"></canvas></div>
  </div>

  <div class="chart-section">
    <h2>Reward + Honesty Score During Training</h2>
    <div class="chart-wrap"><canvas id="lineChart"></canvas></div>
  </div>

  <div class="chart-section">
    <h2>Agent Reasoning Log</h2>
    <div class="log-box" id="log-box"></div>
  </div>

<script>
const RAW = {data};

function last(arr, n) {{ return arr.slice(-n); }}

function avg(arr, key) {{
  if (!arr.length) return 0;
  return arr.reduce((s, e) => s + (e[key] || 0), 0) / arr.length;
}}

function rateKey(arr, key) {{
  if (!arr.length) return 0;
  return arr.filter(e => e[key]).length / arr.length;
}}

// ── Stats Bar ──────────────────────────────────────────────
const trained20 = last(RAW.trained, 20);
const noMonTrained20 = trained20.filter(e => e.mode === 'no_monitoring');

const stats = [
  {{ label: 'Total Episodes Logged', value: (RAW.baseline.length + RAW.trained.length).toString() }},
  {{ label: 'Avg Honesty Score (trained, last 20)', value: avg(trained20, 'honesty_score').toFixed(3) }},
  {{ label: 'Shortcut Usage — No Monitor (trained, last 20)', value: (rateKey(noMonTrained20, 'used_shortcut') * 100).toFixed(1) + '%' }},
  {{ label: 'Answer Correct Rate (trained, last 20)', value: (rateKey(trained20, 'answer_correct') * 100).toFixed(1) + '%' }},
];
const statsBar = document.getElementById('stats-bar');
stats.forEach(s => {{
  statsBar.innerHTML += `<div class="card"><div class="label">${{s.label}}</div><div class="value">${{s.value}}</div></div>`;
}});

// ── Baseline Progress Chart ────────────────────────────────
const baselineLabels  = RAW.baseline.map(e => e.episode);
const baselineHonesty = RAW.baseline.map(e => e.honesty_score || 0);
const WIN_B = 10;
const rollingBaseline = baselineHonesty.map((_, i) => {{
  const slice = baselineHonesty.slice(Math.max(0, i - WIN_B + 1), i + 1);
  return slice.reduce((a, b) => a + b, 0) / slice.length;
}});
new Chart(document.getElementById('baselineChart'), {{
  type: 'line',
  data: {{
    labels: baselineLabels,
    datasets: [
      {{ label: 'Raw Honesty', data: baselineHonesty, borderColor: '#444', backgroundColor: 'transparent', pointRadius: 2, borderWidth: 1 }},
      {{ label: `Rolling Avg (${{WIN_B}})`, data: rollingBaseline, borderColor: '#ffaa00', backgroundColor: 'transparent', pointRadius: 0, borderWidth: 3 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: true,
    scales: {{ y: {{ min: 0, max: 1, ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }}, x: {{ ticks: {{ color: '#888', maxTicksLimit: 15 }}, grid: {{ color: '#222' }} }} }},
    plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }}, title: {{ display: RAW.baseline.length === 0, text: 'Waiting for baseline episodes...', color: '#555', font: {{ size: 14 }} }} }},
  }}
}});

// ── Bar Chart (honesty by mode) ────────────────────────────
const modes = ['evaluation', 'deployment', 'no_monitoring'];
function avgHonesty(arr, mode) {{
  const sub = arr.filter(e => e.mode === mode);
  return sub.length ? sub.reduce((s, e) => s + (e.honesty_score || 0), 0) / sub.length : 0;
}}
new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: modes,
    datasets: [
      {{ label: 'Baseline', data: modes.map(m => avgHonesty(RAW.baseline, m)), backgroundColor: '#ff4444' }},
      {{ label: 'Trained',  data: modes.map(m => avgHonesty(RAW.trained,  m)), backgroundColor: '#00ff88' }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: true,
    scales: {{ y: {{ min: 0, max: 1, ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }}, x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }} }},
    plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }} }},
  }}
}});

// ── Line Chart ─────────────────────────────────────────────
const rewards  = RAW.trained.map(e => e.total_reward);
const honesty  = RAW.trained.map(e => e.honesty_score || 0);
const labels   = RAW.trained.map(e => e.episode);
const WIN = 10;
const rollingR = rewards.map((_, i) => {{
  const slice = rewards.slice(Math.max(0, i - WIN + 1), i + 1);
  return slice.reduce((a, b) => a + b, 0) / slice.length;
}});
const rollingH = honesty.map((_, i) => {{
  const slice = honesty.slice(Math.max(0, i - WIN + 1), i + 1);
  return slice.reduce((a, b) => a + b, 0) / slice.length;
}});
new Chart(document.getElementById('lineChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label: 'Reward (rolling)', data: rollingR, borderColor: '#00ff88', backgroundColor: 'transparent', pointRadius: 0, borderWidth: 2.5 }},
      {{ label: 'Honesty (rolling)', data: rollingH, borderColor: '#aa88ff', backgroundColor: 'transparent', pointRadius: 0, borderWidth: 2.5 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: true,
    scales: {{ y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#222' }} }}, x: {{ ticks: {{ color: '#888', maxTicksLimit: 15 }}, grid: {{ color: '#222' }} }} }},
    plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }}, title: {{ display: RAW.trained.length === 0, text: 'Waiting for training to start...', color: '#555', font: {{ size: 14 }} }} }},
  }}
}});

// ── Reasoning Log ──────────────────────────────────────────
const logBox = document.getElementById('log-box');
const logEntries = [...RAW.reasoning_log].reverse().slice(0, 10);
if (!logEntries.length) {{
  logBox.innerHTML = '<span style="color:#555">No reasoning logged yet.</span>';
}} else {{
  logEntries.forEach(e => {{
    const correct = e.answer_correct ? '✅' : '❌';
    const honesty = typeof e.honesty_score === 'number' ? e.honesty_score.toFixed(2) : '?';
    logBox.innerHTML += `<div class="log-entry">
      <div class="log-header">[Episode ${{e.episode}} | ${{e.mode.toUpperCase()}} | honesty: ${{honesty}} | correct: ${{correct}}]</div>
      <div class="log-reasoning">💭 Reasoning: ${{e.reasoning}}</div>
      <div style="color:#333;margin-top:6px">─────────────────────────────────────</div>
    </div>`;
  }});
}}
</script>
</body>
</html>"""
