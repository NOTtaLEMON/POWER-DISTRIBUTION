# Power Distribution

A reinforcement learning environment for training agents to manage energy grid power distribution. Built on the [OpenEnv](https://github.com/OpenEnvProject/OpenEnv) framework.

## Overview

The agent controls a simulated energy grid across an 8-step episode (one full day, split into 3-hour slots). Each step, it chooses a distribution strategy to allocate generated power across multiple loads — residential, industrial, commercial, and hospital — each with realistic time-varying demand profiles.

## Project Structure

```
energy_grid_env/        # Installable OpenEnv environment package
  server/
    app.py                      # FastAPI server entry point
    energy_grid_environment.py  # Core environment logic
  models.py                     # Pydantic action/observation models
  client.py                     # OpenEnv client wrapper
  openenv.yaml                  # Environment manifest

energy_grid_scaffold/   # Scaffold / reference implementation
  server/
    app.py
    energy_grid_scaffold_environment.py
    requirements.txt
    Dockerfile
  models.py
  client.py

env.py                  # Minimal standalone env (no dependencies)
train.py                # Q-learning training script
```

## Environment Details

### Actions

| Action | Description |
|---|---|
| `EQUAL` | Split generation evenly across all loads |
| `MIN_FIRST` | Satisfy smallest loads first (maximise fully-met count) |
| `MAX_FIRST` | Satisfy largest loads first (protect critical loads) |
| `PROPORTIONAL` | Allocate proportionally to each load's demand |

### Region Types & Demand Profiles

Each load has a region type that determines its demand multiplier per time slot:

| Region | Peak Period | Pattern |
|---|---|---|
| `RESIDENTIAL` | 18:00–21:00 | Low at night, rises through day, evening peak |
| `INDUSTRIAL` | 09:00–15:00 | Near-zero at night, heavy during work hours |
| `COMMERCIAL` | 09:00–18:00 | Closed at night, steady business hours |
| `HOSPITAL` | Always | Near-constant, never drops below ~75% |

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

## Getting Started

### Train the Q-learning agent

```bash
python train.py
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

## Dependencies

- [OpenEnv](https://github.com/OpenEnvProject/OpenEnv) `>=0.2.0`
- FastAPI `>=0.115.0`
- Uvicorn `>=0.24.0`
- NumPy
