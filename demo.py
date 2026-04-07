"""
Interactive demo — enter your grid parameters and see how power is distributed.
The trained Q-learning agent (from train.py) picks the distribution strategy
and battery mode automatically based on your inputs.

Run:  python train.py   # first time, to generate qtable.json
      python demo.py
"""

import json
import os
from energy_grid_env.models import AllocWeight, BatteryMode, RegionType

REGION_LABELS = {
    "R": RegionType.RESIDENTIAL,
    "I": RegionType.INDUSTRIAL,
    "C": RegionType.COMMERCIAL,
    "H": RegionType.HOSPITAL,
}

WEIGHT_MAP = {AllocWeight.NORMAL: 1.0, AllocWeight.HIGH: 2.0}

QTABLE_PATH = os.path.join(os.path.dirname(__file__), "qtable.json")


def load_qtable():
    if not os.path.exists(QTABLE_PATH):
        return None, None
    with open(QTABLE_PATH) as f:
        data = json.load(f)
    # Each action is [res, ind, com, hos, bat] as enum value strings
    action_index = [
        (AllocWeight(r), AllocWeight(i), AllocWeight(c), AllocWeight(h), BatteryMode(b))
        for r, i, c, h, b in data["actions"]
    ]
    table = {}
    for state_str, entry in data["table"].items():
        state = tuple(int(x) for x in state_str.strip("()").split(", "))
        table[state] = entry["q"]
    return table, action_index


def encode_state(total_demand, generation_forecast, battery, num_hospital, demand_trend=0):
    """Mirror of encode_state in train.py."""
    total_supply = generation_forecast + battery
    ratio = total_supply / max(total_demand, 1)
    if ratio < 0.7:
        supply_coverage = 0
    elif ratio < 1.0:
        supply_coverage = 1
    elif ratio < 1.4:
        supply_coverage = 2
    else:
        supply_coverage = 3

    if num_hospital == 0:
        hospital_presence = 0
    elif num_hospital <= 2:
        hospital_presence = 1
    else:
        hospital_presence = 2

    battery_level = 0 if battery <= 2 else (1 if battery <= 6 else 2)

    return (supply_coverage, hospital_presence, demand_trend + 1, battery_level)


# ---------------------------------------------------------------------------
# Distribution logic (mirrors energy_grid_environment.py — no server needed)
# ---------------------------------------------------------------------------

def distribute(type_weights: dict, total: int, demands: list[int], load_types: list) -> list[int]:
    n        = len(demands)
    weighted = [demands[i] * type_weights[load_types[i]] for i in range(n)]
    total_w  = sum(weighted)
    if total_w == 0:
        base, extra = divmod(total, n)
        return [base + (1 if i < extra else 0) for i in range(n)]
    alloc     = [int(total * w / total_w) for w in weighted]
    remainder = total - sum(alloc)
    for i in sorted(range(n), key=lambda i: -weighted[i])[:remainder]:
        alloc[i] += 1
    return alloc


def priority_order(type_weights: dict, demands: list[int], load_types: list) -> list[int]:
    return sorted(range(len(demands)), key=lambda i: -(type_weights[load_types[i]] * demands[i]))


def apply_battery(
    dist: list[int],
    demands: list[int],
    battery: int,
    mode: BatteryMode,
    order: list[int],
) -> tuple[list[int], int]:
    max_draw = {
        BatteryMode.SAVE:     0,
        BatteryMode.MODERATE: battery // 2,
        BatteryMode.SPEND:    battery,
    }[mode]
    budget = max_draw
    used   = 0
    for i in order:
        gap = max(0, demands[i] - dist[i])
        if gap > 0 and budget > 0:
            cover    = min(gap, budget)
            dist[i] += cover
            used    += cover
            budget  -= cover
    return dist, used


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def ask(prompt: str, default=None):
    suffix = f" [{default}]" if default is not None else ""
    raw    = input(f"{prompt}{suffix}: ").strip()
    return raw if raw else (str(default) if default is not None else "")


def ask_int(prompt: str, lo: int, hi: int, default: int = None) -> int:
    while True:
        raw = ask(prompt, default)
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"  Enter a number between {lo} and {hi}.")


def ask_choice(prompt: str, choices: dict, default: str = None) -> str:
    while True:
        raw = ask(prompt, default).upper()
        if raw in choices:
            return raw
        print(f"  Enter one of: {', '.join(choices.keys())}")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    SEP = "─" * 52
    print(f"\n{SEP}")
    print("  Energy Grid Distribution Demo")
    print(SEP)

    # Load trained Q-table
    qtable, action_index = load_qtable()
    if qtable is None:
        print("\n  No trained model found. Run  python train.py  first.\n")
        return

    # --- Loads ---
    n = ask_int("\nHow many loads? (1–10)", 1, 10, default=3)
    loads = []
    print()
    for idx in range(n):
        print(f"  Load {idx + 1}:")
        rtype = ask_choice(
            "    Region type  R=Residential  I=Industrial  C=Commercial  H=Hospital",
            REGION_LABELS,
            default="R",
        )
        demand = ask_int("    Demand (units)", 1, 50, default=5)
        loads.append((REGION_LABELS[rtype], demand))

    # --- Supply ---
    print()
    generation = ask_int("Generation (power produced this step, units)", 0, 100, default=10)
    battery    = ask_int("Battery charge (stored energy, 0–10)", 0, 10, default=5)

    # --- Agent picks strategy + battery mode ---
    demands      = [d for _, d in loads]
    types        = [t for t, _ in loads]
    num_hospital = sum(1 for t in types if t == RegionType.HOSPITAL)
    state        = encode_state(sum(demands), generation, battery, num_hospital)

    if state in qtable:
        action_idx = int(max(range(len(qtable[state])), key=lambda i: qtable[state][i]))
    else:
        # Unseen state — fall back to NORMAL weights + SPEND
        action_idx = next(
            i for i, a in enumerate(action_index)
            if all(w == AllocWeight.NORMAL for w in a[:4]) and a[4] == BatteryMode.SPEND
        )

    res_w, ind_w, com_w, hos_w, bmode = action_index[action_idx]
    type_weights = {
        RegionType.RESIDENTIAL: WEIGHT_MAP[res_w],
        RegionType.INDUSTRIAL:  WEIGHT_MAP[ind_w],
        RegionType.COMMERCIAL:  WEIGHT_MAP[com_w],
        RegionType.HOSPITAL:    WEIGHT_MAP[hos_w],
    }

    # --- Compute ---
    dist  = distribute(type_weights, generation, demands, types)
    order = priority_order(type_weights, demands, types)
    dist, battery_used = apply_battery(dist, demands, battery, bmode, order)

    gen_used    = sum(min(demands[i], dist[i]) for i in range(n))
    gen_surplus = max(0, generation - gen_used)
    new_battery = min(10, battery - battery_used + gen_surplus)

    delivery_ratios = [min(dist[i], demands[i]) / demands[i] for i in range(n)]
    avg_delivery    = sum(delivery_ratios) / n
    reward          = 2.0 * avg_delivery - 1.0 + 0.1 * (new_battery / 10)

    # --- Output ---
    print(f"\n{SEP}")
    print(f"  Agent weights →  RES:{res_w.value}  IND:{ind_w.value}  COM:{com_w.value}  HOS:{hos_w.value}  | battery: {bmode.value}")
    print(SEP)
    print(f"  {'#':<4} {'Type':<14} {'Demand':>7} {'Supplied':>9} {'Met':>8}  {'Status'}")
    print(f"  {'─'*4} {'─'*14} {'─'*7} {'─'*9} {'─'*8}  {'─'*12}")
    for i in range(n):
        supplied = dist[i]
        met_pct  = delivery_ratios[i] * 100
        status   = "FULL" if supplied >= demands[i] else f"SHORT {demands[i]-supplied} unit{'s' if demands[i]-supplied != 1 else ''}"
        print(f"  {i+1:<4} {types[i].value:<14} {demands[i]:>7} {supplied:>9} {met_pct:>7.0f}%  {status}")

    print(f"\n  Total demand   : {sum(demands)}")
    print(f"  Total supplied : {sum(dist)}")
    print(f"  Generation     : {generation}  (used: {gen_used}, surplus→battery: {gen_surplus})")
    print(f"  Battery        : {battery} → {new_battery}  (drew: {battery_used})")
    print(f"  Avg delivery   : {avg_delivery*100:.1f}%")
    print(f"  Step reward    : {reward:+.3f}")
    print(SEP)


if __name__ == "__main__":
    main()
