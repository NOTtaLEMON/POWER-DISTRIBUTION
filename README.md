# Power Distribution

A reinforcement learning environment for training agents to manage energy grid power distribution. Built on the [OpenEnv](https://github.com/OpenEnvProject/OpenEnv) framework.

## Overview

The agent controls a simulated energy grid across an 8-step episode (one full day, split into 3-hour slots). Each step, it chooses how to distribute generated power across multiple loads — residential, industrial, commercial, and hospital — and how aggressively to draw from the battery. Demand varies with time of day, and generation is only visible as a noisy forecast (±20%).

## Project Structure

```
energy_grid_env/        # Installable OpenEnv environment package
  server/
    app.py                      # FastAPI server entry point
    energy_grid_environment.py  # Core environment logic
    Dockerfile                  # Container build for HF Space / eval harness
  models.py                     # Pydantic action/observation models (action + observation types)
  client.py                     # OpenEnv HTTP client wrapper
  openenv.yaml                  # Environment manifest — spec, actions, tasks, graders
  pyproject.toml                # Package metadata and pip dependencies

energy_grid_scaffold/   # Scaffold / reference implementation (mirrors energy_grid_env)
  server/
    app.py
    energy_grid_scaffold_environment.py
    requirements.txt
    Dockerfile
  models.py
  client.py
  pyproject.toml

env.py                  # Minimal standalone env — no server, no dependencies, for fast local use
train.py                # Tabular Q-learning training script
inference.py            # LLM agent inference script (competition submission entry point)
demo.py                 # Interactive demo — manually step through the environment
qtable.json             # Saved Q-table from the last training run
```

## Environment Details

### Actions

Each step the agent chooses two things:

**Distribution strategy** — how to split generated power across loads:

| Action | Description |
|---|---|
| `EQUAL` | Split generation evenly across all loads |
| `MIN_FIRST` | Satisfy smallest loads first (maximise fully-met count) |
| `MAX_FIRST` | Satisfy largest loads first (protect critical loads) |
| `PROPORTIONAL` | Allocate proportionally to each load's demand |

**Battery mode** — how aggressively to draw stored energy to cover gaps:

| Mode | Description |
|---|---|
| `SAVE` | Don't draw battery this step — reserve for later peaks |
| `MODERATE` | Draw up to half of remaining battery to cover gaps |
| `SPEND` | Draw as much battery as needed to cover all gaps |

### Region Types & Demand Profiles

Each load has a region type that determines its demand multiplier per time slot:

| Region | Peak Period | Pattern |
|---|---|---|
| `RESIDENTIAL` | 18:00–21:00 | Low at night, rises through day, evening peak |
| `INDUSTRIAL` | 09:00–15:00 | Near-zero at night, heavy during work hours |
| `COMMERCIAL` | 09:00–18:00 | Closed at night, steady business hours |
| `HOSPITAL` | Always | Near-constant, never drops below ~75% |

~15% of loads are **anomalous** — they ignore their region profile and draw from an unpredictable range, forcing the agent to react to observed demand rather than memorised patterns.

### Generation Forecast

The agent sees `generation_forecast` — a noisy ±20% estimate of actual generation. True generation is hidden. The agent must account for this uncertainty in its distribution and battery decisions.

### Reward

```
reward = 2 × (avg delivery ratio) - 1 + 0.1 × (battery / 10)
```

Where `delivery ratio = min(supplied, demanded) / demanded` averaged over all loads.

| Score (per episode, max ~8.8) | Rating |
|---|---|
| ≥ 7.0 | Excellent |
| 4.0 – 7.0 | Good |
| 0.0 – 4.0 | Average |
| -4.0 – 0.0 | Poor |
| < -4.0 | Bad |

### Task Graders (0.0 – 1.0)

Three graders score each step independently, reported in `task_scores`:

| Task | Description |
|---|---|
| `delivery` | Average fraction of total demand met across all loads |
| `hospital_coverage` | Average fraction of hospital/critical demand met |
| `battery_management` | Fraction of battery remaining — rewards conservative use |

## Getting Started

### Train the Q-learning agent

```bash
python train.py
```

### Run the LLM inference script

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=openai/gpt-4o-mini
export HF_TOKEN=your_token_here
python inference.py
```

### Run the environment server

Install dependencies:
```bash
pip install openenv[core] fastapi uvicorn
```

Start the server:
```bash
cd energy_grid_env
uvicorn server.app:app --reload
```

### Build the Docker image

```bash
docker build -f energy_grid_env/server/Dockerfile -t energy-grid:latest energy_grid_env
```

## Dependencies

- [OpenEnv](https://github.com/OpenEnvProject/OpenEnv) `>=0.2.0`
- FastAPI `>=0.115.0`
- Uvicorn `>=0.24.0`
- NumPy
- OpenAI Python client (for `inference.py`)

