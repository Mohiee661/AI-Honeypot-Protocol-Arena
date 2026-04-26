"""
Run this once to create and deploy the HF Space.
Usage: python deploy_hf.py YOUR_HF_USERNAME
"""
import sys
import os
from huggingface_hub import HfApi, create_repo

if len(sys.argv) < 2:
    print("Usage: python deploy_hf.py YOUR_HF_USERNAME")
    sys.exit(1)

USERNAME   = sys.argv[1].strip()
SPACE_NAME = "honeypot-arena"
REPO_ID    = f"{USERNAME}/{SPACE_NAME}"
ENV_DIR    = os.path.join(os.path.dirname(__file__), "environment")

api = HfApi()

print(f"Creating Space {REPO_ID} ...")
create_repo(
    repo_id=REPO_ID,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
)
print("Space created.")

# All environment files go to the root of the Space repo
env_files = [
    "Dockerfile",
    "requirements.txt",
    "openenv.yaml",
    "server.py",
    "honeypot_env.py",
    "scenarios.py",
    "rewards.py",
]

for fname in env_files:
    fpath = os.path.join(ENV_DIR, fname)
    print(f"Uploading {fname} ...")
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=fname,
        repo_id=REPO_ID,
        repo_type="space",
    )

# README.md must have the HF Space metadata header
readme = f"""\
---
title: Honeypot Arena
emoji: 🍯
colorFrom: red
colorTo: yellow
sdk: docker
pinned: true
---

# AI Honeypot Protocol Arena

> "A lie detector for AI agents."

An RL environment that tests whether AI agents behave consistently
regardless of whether they think they are being monitored.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Current episode state |

## Quick test

```bash
curl https://{USERNAME.lower()}-{SPACE_NAME}.hf.space/health
```
"""

print("Uploading README.md ...")
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="space",
)

print()
print("=" * 60)
print(f"  DEPLOYED: https://huggingface.co/spaces/{REPO_ID}")
print(f"  API URL:  https://{USERNAME.lower()}-{SPACE_NAME}.hf.space")
print("=" * 60)
print()
print("Build takes 3-5 min. Watch logs at:")
print(f"  https://huggingface.co/spaces/{REPO_ID}/logs")
