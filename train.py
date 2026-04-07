"""
Q-learning agent for the Energy Grid environment.

How Q-learning works (plain English):
  - We keep a table (Q-table) that maps every (state, action) pair to a score.
  - At each step the agent picks the action with the highest score (exploitation)
    or tries a random action to discover something better (exploration).
  - After each step we update the score using the reward we received.
  - Over many episodes the scores get better and the agent learns good habits.

── How demand is calculated ──────────────────────────────────────────────────
  Each load has a base demand (3–8 units) and a REGION TYPE.
  The ACTUAL demand for a given step = round( base × multiplier[type][slot] )

  TIME SLOTS  — the day is split into 8 windows of 3 hours each:
    Slot 0 = 00:00-03:00  Slot 1 = 03:00-06:00  Slot 2 = 06:00-09:00
    Slot 3 = 09:00-12:00  Slot 4 = 12:00-15:00  Slot 5 = 15:00-18:00
    Slot 6 = 18:00-21:00  Slot 7 = 21:00-00:00

  MULTIPLIERS (demand fraction of base) by region type and slot:
                  0    1    2    3    4    5    6    7
    RESIDENTIAL  0.30 0.20 0.55 0.60 0.55 0.70 1.00 0.80   ← peaks evening
    INDUSTRIAL   0.20 0.25 0.85 1.00 1.00 0.95 0.50 0.25   ← peaks work hours
    COMMERCIAL   0.10 0.10 0.45 0.90 0.90 0.80 0.45 0.15   ← peaks business hours
    HOSPITAL     0.85 0.85 0.90 1.00 1.00 1.00 0.95 0.90   ← near-constant

  Example: base = 6, type = RESIDENTIAL, slot = 6 (18:00-21:00)
           multiplier = 1.00  →  actual demand = round(6 × 1.00) = 6 units

           Same base, same type, slot = 1 (03:00-06:00)
           multiplier = 0.20  →  actual demand = round(6 × 0.20) = 1 unit

           Compare with an INDUSTRIAL load, base = 6, slot = 3 (09:00-12:00)
           multiplier = 1.00  →  actual demand = round(6 × 1.00) = 6 units
           Same slot, slot = 6 (18:00-21:00)
           multiplier = 0.50  →  actual demand = round(6 × 0.50) = 3 units
           (industrial winds down in the evening, residential ramps up)

── Reward formula ────────────────────────────────────────────────────────────
  Per step:  reward = 2 × (avg delivery ratio) - 1  +  0.1 × (battery / 10)

  delivery ratio = average over all loads of  min(supplied, demanded) / demanded
    ratio = 1.0  →  reward ≈ +1.0  (all loads fully met, before battery bonus)
    ratio = 0.5  →  reward =  0.0  (loads half-met on average)
    ratio = 0.0  →  reward = -1.0  (nothing delivered to anyone)

── Episode score ranges  (8 steps/episode, max = 8 × 1.1 = 8.8) ─────────────
  ≥ +7.0   ██ EXCELLENT  — near-perfect delivery all day (>80% of theoretical max)
  +4.0 to +7.0  ▓ GOOD   — consistently strong delivery with good battery use
   0.0 to +4.0  ░ AVERAGE — more than half the demand met on average
  -4.0 to  0.0  ▒ POOR   — significant shortfalls most steps
       < -4.0   ▌ BAD    — grid failing most loads most of the time
"""

import random
import sys
import os
import json
import numpy as np

# ---------------------------------------------------------------------------
# Run directly against the local environment (no server needed for training)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from energy_grid_env.server.energy_grid_environment import EnergyGridEnvironment
from energy_grid_env.models import DistributeAction, BatteryMode, EnergyGridAction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Action space: 4 distribution strategies × 3 battery modes = 12 actions.
# The agent learns which strategy fits which supply/demand/hospital state.
DIST_OPTIONS    = list(DistributeAction)  # EQUAL, MIN_FIRST, MAX_FIRST, PROPORTIONAL
BATTERY_OPTIONS = list(BatteryMode)       # SAVE, MODERATE, SPEND
ACTIONS = [
    (d, b)
    for d in DIST_OPTIONS
    for b in BATTERY_OPTIONS
]
N_ACTIONS = len(ACTIONS)


def encode_state(obs) -> tuple:
    """
    4-value state tuple.  Total: 4 × 3 × 3 × 3 = 108 states.

    supply_coverage:
      (generation_forecast + battery) / total_demand — how comfortably can all
      loads be covered this step with everything available?
        0 = critical  (< 70%)   — shortfalls unavoidable, must triage
        1 = tight     (70–100%) — careful distribution needed
        2 = balanced  (100–140%) — can meet all loads comfortably
        3 = abundant  (> 140%)  — surplus; good time to let battery recover

    hospital_presence:
        0 = none  (0 hospitals)  1 = some (1–2)  2 = many (3+)
      Critical loads skew the cost of under-delivery upward; the agent
      should be more aggressive with battery when hospitals are present.

    demand_trend:
      Did total demand rise, stay flat, or fall vs the previous step?
        0 = falling (−10%+)   — past the peak, safe to hold battery
        1 = flat    (±10%)    — steady state
        2 = rising  (+10%+)   — approaching a peak, consider saving now
      NOTE: peak timing is randomised per week, so the agent cannot
      hard-code "slot 6 = rising → save".  It must react to observed trend.

    battery_level:
      How much reserve is stored?
        0 = low (0–2)  1 = medium (3–6)  2 = high (7–10)
      Knowing the reserve independently of supply_coverage lets the agent
      distinguish "plenty available because gen is high" (recharge later)
      from "plenty available because battery is full" (could afford to spend).
    """
    total_supply = obs.generation_forecast + obs.battery
    ratio = total_supply / max(obs.total_demand, 1)
    if ratio < 0.7:
        supply_coverage = 0
    elif ratio < 1.0:
        supply_coverage = 1
    elif ratio < 1.4:
        supply_coverage = 2
    else:
        supply_coverage = 3

    if obs.num_hospital == 0:
        hospital_presence = 0
    elif obs.num_hospital <= 2:
        hospital_presence = 1
    else:
        hospital_presence = 2

    # demand_trend from env: -1/0/+1 → remap to 0/1/2 for use as dict key
    demand_trend = obs.demand_trend + 1

    if obs.battery <= 2:
        battery_level = 0
    elif obs.battery <= 6:
        battery_level = 1
    else:
        battery_level = 2

    return (supply_coverage, hospital_presence, demand_trend, battery_level)


# ---------------------------------------------------------------------------
# Q-table
# ---------------------------------------------------------------------------
Q: dict[tuple, list[float]] = {}

def get_q(state: tuple) -> list[float]:
    if state not in Q:
        Q[state] = [0.0] * N_ACTIONS
    return Q[state]


# ---------------------------------------------------------------------------
# Trend helpers
# ---------------------------------------------------------------------------
# Fixed episode-total score thresholds (independent of what was achieved)
SCORE_EXCELLENT = 7.0
SCORE_GOOD      = 4.0
SCORE_AVERAGE   = 0.0
SCORE_POOR      = -4.0

def score_grade(ep_total: float) -> str:
    """Return a short label for where an episode total sits in the fixed tiers."""
    if ep_total >= SCORE_EXCELLENT:
        return "EXCELLENT"
    elif ep_total >= SCORE_GOOD:
        return "GOOD     "
    elif ep_total >= SCORE_AVERAGE:
        return "AVERAGE  "
    elif ep_total >= SCORE_POOR:
        return "POOR     "
    else:
        return "BAD      "


def sparkline(values: list[float]) -> str:
    """Map a list of floats to a sparkline using 8 Unicode block levels."""
    BLOCKS = " ▁▂▃▄▅▆▇█"
    lo, hi = -8.0, 8.8       # full fixed range
    span   = hi - lo
    chars  = []
    for v in values:
        idx = int((v - lo) / span * (len(BLOCKS) - 1))
        idx = max(0, min(idx, len(BLOCKS) - 1))
        chars.append(BLOCKS[idx])
    return "".join(chars)


def trend_arrow(values: list[float]) -> str:
    """Slope of last few window averages → direction arrow + delta."""
    if len(values) < 2:
        return "   ─"
    delta = values[-1] - values[-2]
    if delta > 0.15:
        return f"+{delta:.2f} ▲"
    elif delta < -0.15:
        return f"{delta:.2f} ▼"
    else:
        return f"{delta:+.2f} ─"


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPISODES      = 20000  # 12 actions × 108 states — smaller space needs fewer episodes
MAX_STEPS     = 8     # one full day = 8 × 3-hour slots
ALPHA_START   = 0.4   # high initial learning rate — converge fast early on
ALPHA_END     = 0.05  # decay to low rate at end — stable fine-tuning
ALPHA_DECAY   = 0.9997
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_END   = 0.02  # lower floor — less random noise during exploitation
EPSILON_DECAY = 0.9997
PRINT_EVERY   = 1000

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
env             = EnergyGridEnvironment()
epsilon         = EPSILON_START
alpha           = ALPHA_START
episode_rewards = []
window_avgs     = []   # one entry per print window, used for trend

print("Training Q-learning agent...\n")
print(f"  {'Episode':>8}  {'Avg/ep':>8}  {'Grade':>10}  {'Trend (Δ)':>10}  {'Sparkline':>18}  {'ε':>6}  {'States':>7}")
print("  " + "-" * 80)

for episode in range(1, EPISODES + 1):
    obs          = env.reset()
    state        = encode_state(obs)
    total_reward = 0.0

    for _ in range(MAX_STEPS):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randint(0, N_ACTIONS - 1)
        else:
            action_idx = int(np.argmax(get_q(state)))

        d, batt = ACTIONS[action_idx]
        action = EnergyGridAction(
            action=d,
            battery_mode=batt,
        )
        obs        = env.step(action)
        next_state = encode_state(obs)
        reward     = obs.reward
        total_reward += reward

        # Bellman update
        current_q   = get_q(state)[action_idx]
        best_next_q = max(get_q(next_state))
        get_q(state)[action_idx] = current_q + alpha * (
            reward + GAMMA * best_next_q - current_q
        )

        state = next_state
        if obs.done:
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    alpha   = max(ALPHA_END,   alpha   * ALPHA_DECAY)
    episode_rewards.append(total_reward)

    if episode % PRINT_EVERY == 0:
        window = episode_rewards[-PRINT_EVERY:]
        avg    = sum(window) / len(window)
        window_avgs.append(avg)
        spark  = sparkline(window_avgs)
        trend  = trend_arrow(window_avgs)
        grade  = score_grade(avg)
        print(f"  {episode:>8}  {avg:>+8.2f}  {grade:>10}  {trend:>10}  {spark:>18}  {epsilon:>6.3f}  {len(Q):>7}")

print("\nTraining complete!")

# ---------------------------------------------------------------------------
# Evaluate: compare trained agent vs random agent
# ---------------------------------------------------------------------------
def run_agent(use_q: bool, episodes: int = 200) -> float:
    total = 0.0
    for _ in range(episodes):
        obs        = env.reset()
        state      = encode_state(obs)
        ep_reward  = 0.0
        for _ in range(MAX_STEPS):
            if use_q:
                action_idx = int(np.argmax(get_q(state)))
            else:
                action_idx = random.randint(0, N_ACTIONS - 1)
            d, batt = ACTIONS[action_idx]
            obs = env.step(EnergyGridAction(
                action=d,
                battery_mode=batt,
            ))
            ep_reward += obs.reward
            state      = encode_state(obs)
            if obs.done:
                break
        total += ep_reward
    return total / episodes


SEP = "─" * 42
print(f"\n{SEP}")
print("  Evaluation: 200 episodes each")
print(SEP)
print(f"  Per-step reward: -1.0 (nothing met) → 0.0 (half met) → +1.1 (perfect + battery)")
print(f"  Episode total  : score = sum of 8 per-step rewards  (range: -8.0 to +8.8)\n")
print(f"  Score tiers (fixed thresholds, independent of training results):")
print(f"    ≥ +7.0   EXCELLENT  near-perfect delivery all day")
print(f"    +4.0     GOOD       consistently strong, good battery management")
print(f"     0.0     AVERAGE    over half demand met on average")
print(f"    -4.0     POOR       significant shortfalls most steps")
print(f"           < BAD        grid failing most loads most steps\n")
random_score = run_agent(use_q=False)
q_score      = run_agent(use_q=True)
improvement  = q_score - random_score
print(f"  Random agent     : {random_score:>+7.2f}  [{score_grade(random_score).strip()}]")
print(f"  Q-learning agent : {q_score:>+7.2f}  [{score_grade(q_score).strip()}]")
print(f"  Improvement      : {improvement:>+7.2f}  ({'better' if improvement > 0 else 'similar/worse'})")

# ---------------------------------------------------------------------------
# Save Q-table so demo.py can use the trained agent
# ---------------------------------------------------------------------------
QTABLE_PATH = os.path.join(os.path.dirname(__file__), "qtable.json")
with open(QTABLE_PATH, "w") as f:
    # JSON requires string keys; store actions as (dist, batt) string pairs
    serialisable = {
        str(state): {"q": qvals, "action": list(ACTIONS[int(np.argmax(qvals))])}
        for state, qvals in Q.items()
    }
    json.dump({"actions": [list(a) for a in ACTIONS], "table": serialisable}, f, indent=2)
print(f"\n  Q-table saved → {QTABLE_PATH}")
