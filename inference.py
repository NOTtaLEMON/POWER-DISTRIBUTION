#!/usr/bin/env python3
"""LLM inference script for the Energy Grid environment.

An LLM agent plays the energy grid environment, choosing a power distribution
strategy each 3-hour slot across an 8-step episode (one full day).

Usage
-----
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=openai/gpt-4o-mini
    export HF_TOKEN=your_token_here
    python inference.py

The script emits structured stdout logs in [START] / [STEP] / [END] format
that the evaluation harness uses to score each episode.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — all sourced from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "openai/gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", "3"))
MAX_STEPS    = 8   # one full day = 8 × 3-hour slots

ACTIONS = ["EQUAL", "MIN_FIRST", "MAX_FIRST", "PROPORTIONAL"]

SYSTEM_PROMPT = """\
You are an energy grid operator managing power distribution for one day.
Each 3-hour time slot you must choose a distribution strategy.

Strategies:
- EQUAL        — split generation evenly across all loads
- MIN_FIRST    — satisfy smallest loads first (maximise fully-met count)
- MAX_FIRST    — satisfy largest loads first (protect critical loads)
- PROPORTIONAL — allocate proportionally to each load's demand

Your goal is to maximise delivery to all loads while conserving battery for peak periods.
Hospitals have near-constant high demand and must be prioritised.

Respond with ONLY one of: EQUAL, MIN_FIRST, MAX_FIRST, PROPORTIONAL
"""

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=HF_TOKEN or "EMPTY",
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset() -> dict[str, Any]:
    resp = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: str) -> dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

def obs_to_prompt(obs: dict[str, Any]) -> str:
    supply = obs.get("generation_forecast", 0) + obs.get("battery", 0)
    return (
        f"Time slot {obs.get('time_slot', '?')}: {obs.get('time_label', '')}\n"
        f"Active loads: {obs.get('num_loads', '?')} "
        f"(residential={obs.get('num_residential',0)}, "
        f"industrial={obs.get('num_industrial',0)}, "
        f"commercial={obs.get('num_commercial',0)}, "
        f"hospital={obs.get('num_hospital',0)})\n"
        f"Total demand: {obs.get('total_demand', '?')} units "
        f"(min={obs.get('min_load','?')}, max={obs.get('max_load','?')})\n"
        f"Generation forecast: {obs.get('generation_forecast', '?')} units\n"
        f"Battery: {obs.get('battery', '?')} / 10 units\n"
        f"Total available supply: {supply} units\n"
        f"Last reward: {obs.get('reward', 0):.3f}\n"
        f"\nChoose one strategy: EQUAL, MIN_FIRST, MAX_FIRST, PROPORTIONAL"
    )


def choose_action(obs: dict[str, Any]) -> str:
    prompt = obs_to_prompt(obs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=16,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip().upper()
        # Extract first matching action keyword
        for action in ACTIONS:
            if action in text:
                return action
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}, defaulting to PROPORTIONAL", file=sys.stderr)
    return "PROPORTIONAL"

# ---------------------------------------------------------------------------
# Structured logging helpers
# ---------------------------------------------------------------------------

def log(tag: str, data: dict[str, Any]) -> None:
    """Emit a structured log line consumed by the evaluation harness."""
    print(f"[{tag}] {json.dumps(data)}", flush=True)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_episode(episode_num: int) -> dict[str, Any]:
    obs_data = env_reset()
    # unwrap observation if nested
    obs = obs_data.get("observation", obs_data)

    log("START", {
        "episode":   episode_num,
        "env":       "energy_grid",
        "model":     MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    total_reward  = 0.0
    step_rewards  = []
    task_scores_accum: dict[str, list[float]] = {
        "delivery": [], "hospital_coverage": [], "battery_management": []
    }

    for step in range(1, MAX_STEPS + 1):
        action = choose_action(obs)
        result = env_step(action)

        next_obs = result.get("observation", result)
        reward   = float(result.get("reward", 0.0))
        done     = bool(result.get("done", False))

        # Compute per-step task scores (all 0.0–1.0)
        task_scores = result.get("task_scores", {})
        for key in task_scores_accum:
            if key in task_scores:
                task_scores_accum[key].append(float(task_scores[key]))

        total_reward += reward
        step_rewards.append(reward)

        log("STEP", {
            "episode":     episode_num,
            "step":        step,
            "action":      action,
            "reward":      round(reward, 4),
            "done":        done,
            "task_scores": {k: round(v, 4) for k, v in task_scores.items()},
            "observation": {
                "time_label":           next_obs.get("time_label", ""),
                "total_demand":         next_obs.get("total_demand", 0),
                "generation_forecast":  next_obs.get("generation_forecast", 0),
                "battery":              next_obs.get("battery", 0),
                "num_hospital":         next_obs.get("num_hospital", 0),
            },
        })

        obs = next_obs
        if done:
            break

    # Average task scores across steps
    avg_task_scores = {
        k: round(sum(v) / len(v), 4) if v else 0.0
        for k, v in task_scores_accum.items()
    }

    log("END", {
        "episode":     episode_num,
        "total_reward": round(total_reward, 4),
        "steps":        len(step_rewards),
        "task_scores":  avg_task_scores,
    })

    return {"total_reward": total_reward, "task_scores": avg_task_scores}


def main() -> None:
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set — API calls may fail", file=sys.stderr)

    results = []
    for ep in range(1, MAX_EPISODES + 1):
        result = run_episode(ep)
        results.append(result)

    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    agg_tasks: dict[str, float] = {}
    for key in ["delivery", "hospital_coverage", "battery_management"]:
        vals = [r["task_scores"].get(key, 0.0) for r in results]
        agg_tasks[key] = round(sum(vals) / len(vals), 4)

    print("\n" + "=" * 50, flush=True)
    print(f"Episodes:          {MAX_EPISODES}", flush=True)
    print(f"Avg total reward:  {avg_reward:.3f}  (range: -8.0 to +8.8)", flush=True)
    print(f"Task scores (0–1): {json.dumps(agg_tasks, indent=2)}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
