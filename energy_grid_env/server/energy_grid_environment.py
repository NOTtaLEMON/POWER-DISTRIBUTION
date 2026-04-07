from __future__ import annotations

import random
import uuid

try:
    from openenv.core.env_server import Environment
    from openenv.core.env_server.types import State
except ImportError:
    from core.env_server import Environment
    from core.env_server.types import State

from ..models import AllocWeight, BatteryMode, EnergyGridAction, EnergyGridObservation, RegionType

# ---------------------------------------------------------------------------
# Time slots: each slot = 3 hours, 8 slots cover one full day
# ---------------------------------------------------------------------------
TIME_LABELS = [
    "00:00-03:00",  # 0 — deep night
    "03:00-06:00",  # 1 — early morning
    "06:00-09:00",  # 2 — morning rush
    "09:00-12:00",  # 3 — midday
    "12:00-15:00",  # 4 — afternoon
    "15:00-18:00",  # 5 — late afternoon
    "18:00-21:00",  # 6 — evening peak
    "21:00-00:00",  # 7 — night wind-down
]

# Demand profiles: (min_multiplier, max_multiplier) per time slot.
# actual_demand = round( base × uniform(min, max) ), minimum 1.
#
# Chosen ranges reflect real-world patterns:
#   Residential  — very low at night, slow rise through the day, evening peak.
#   Industrial   — near-zero at night, heavy during work hours, quiet by evening.
#   Commercial   — closed at night, steady business hours, drops after 6pm.
#   Hospital     — always on; small dip at night but never near zero.
#
#                      0              1              2              3
#                  00:00-03:00    03:00-06:00    06:00-09:00    09:00-12:00
DEMAND_PROFILES: dict[str, list[tuple[float, float]]] = {
    RegionType.RESIDENTIAL: [
        (0.20, 0.35), (0.15, 0.25), (0.35, 0.60), (0.45, 0.65),  #  0-3
        (0.45, 0.65), (0.60, 0.80), (0.85, 1.05), (0.55, 0.75),  #  4-7
    ],
    RegionType.INDUSTRIAL: [
        (0.15, 0.25), (0.15, 0.25), (0.70, 0.95), (0.90, 1.10),  #  0-3
        (0.90, 1.10), (0.80, 1.00), (0.35, 0.55), (0.15, 0.30),  #  4-7
    ],
    RegionType.COMMERCIAL: [
        (0.05, 0.15), (0.05, 0.10), (0.30, 0.55), (0.80, 1.00),  #  0-3
        (0.80, 1.00), (0.65, 0.85), (0.30, 0.50), (0.10, 0.20),  #  4-7
    ],
    RegionType.HOSPITAL: [
        (0.75, 0.90), (0.75, 0.90), (0.80, 0.95), (0.90, 1.05),  #  0-3
        (0.90, 1.05), (0.90, 1.05), (0.85, 1.00), (0.80, 0.95),  #  4-7
    ],
}

# Anomalous loads ignore their region profile for the whole episode.
# They draw demand from a wide flat range — simulating unpredictable
# consumers (e.g. a residential area running heavy appliances at 3am,
# or an industrial plant taking an unscheduled midday shutdown).
# The agent must learn to respond to actual observed demand levels
# rather than relying solely on time_slot patterns.
ANOMALY_LOAD_PROB = 0.15              # per-load, per-episode
ANOMALY_RANGE     = (0.20, 1.05)     # effectively "could be anything"

# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------
MAX_STEPS   = 8    # one full day per episode (8 × 3-hour slots)
MAX_BATTERY = 10
MIN_LOADS   = 3
MAX_LOADS   = 10

PROB_LOAD_REMOVED = 0.05
PROB_LOAD_ADDED   = 0.07

ALL_TYPES = list(RegionType)


class EnergyGridEnvironment(Environment):
    """
    Energy grid with dynamic loads, time-varying demand, and anomalies.

    Each episode = one 24-hour day (8 steps, one per 3-hour slot).
    Each load has a region type and a base demand.  Actual demand per step
    is sampled from a (min, max) range that changes by region type and slot:

      Residential  very low at night (0.15–0.35×), peaks 18:00-21:00 (0.85–1.05×)
      Industrial   near zero at night, peaks 09:00-18:00 (0.90–1.10×)
      Commercial   minimal outside business hours, steady 09:00-18:00 (0.80–1.00×)
      Hospital     always on; slight night dip but never below 0.75×

    ANOMALOUS LOADS (~15% of loads per episode): ignore their region profile
    and draw demand from a flat wide range (0.20–1.05×) every step.  This
    forces the agent to react to actual observed demand rather than just
    memorising “slot 6 = high for residential”.  The agent cannot see the
    anomalous flag — it must infer from gen_coverage and hospital_presence.
    """

    def __init__(self):
        super().__init__()
        # Each load stored as [base_demand, region_type, is_anomalous]
        self.loads: list[list] = []
        self.generation = 0
        self.battery    = 5
        self.time_slot  = 0
        self._state     = State(episode_id=str(uuid.uuid4()), step_count=0)
        # Weekly demand-profile rotation: every 7 episodes the peak window shifts
        # randomly so the agent can't hard-code "slot 6 = high demand = spend".
        self._episode_count  = 0
        self._week_offset    = random.randint(0, 7)
        self.prev_total_demand: int = 0

    # ------------------------------------------------------------------
    # OpenEnv required methods
    # ------------------------------------------------------------------

    def reset(self) -> EnergyGridObservation:
        # Rotate demand peak window every 7 episodes (one "week")
        self._episode_count += 1
        if self._episode_count % 7 == 0:
            self._week_offset = random.randint(0, 7)

        n = random.randint(MIN_LOADS, MAX_LOADS)
        self.loads      = [
            [random.randint(3, 8), random.choice(ALL_TYPES), random.random() < ANOMALY_LOAD_PROB]
            for _ in range(n)
        ]
        self.generation = random.randint(5, 15)
        self.battery    = 5
        self.time_slot  = 0
        self._state     = State(episode_id=str(uuid.uuid4()), step_count=0)
        demands = [self._current_demand(i) for i in range(n)]
        self.prev_total_demand = sum(demands)
        return self._make_obs(demands, reward=0.0, done=False, message="Day started. Slot 0: 00:00-03:00", generation_forecast=self._noisy_forecast(self.generation), demand_trend=0)

    def step(self, action: EnergyGridAction) -> EnergyGridObservation:
        self._state.step_count += 1
        n = len(self.loads)

        # Current demand for each load (base × time multiplier)
        demands = [self._current_demand(i) for i in range(n)]

        # --- 1. Distribute generation using learned per-type weights ---
        dist = self._distribute(action, self.generation, demands)

        # --- 2. Battery covers remaining gaps — respecting chosen battery mode ---
        order = self._priority_order(action, demands)
        max_battery_draw = {
            BatteryMode.SAVE:     0,
            BatteryMode.MODERATE: self.battery // 2,
            BatteryMode.SPEND:    self.battery,
        }[action.battery_mode]
        battery_budget = max_battery_draw
        battery_used   = 0
        for i in order:
            gap = max(0, demands[i] - dist[i])
            if gap > 0 and battery_budget > 0:
                cover         = min(gap, battery_budget)
                dist[i]      += cover
                battery_used += cover
                battery_budget -= cover
        self.battery = max(0, self.battery - battery_used)

        # --- 3. Surplus generation recharges battery ---
        gen_used    = sum(min(demands[i], dist[i]) for i in range(n))
        gen_surplus = max(0, self.generation - gen_used)
        self.battery = min(MAX_BATTERY, self.battery + gen_surplus)

        # --- 4. Reward: continuous delivery ratio ---
        delivery_ratio = sum(min(dist[i], demands[i]) / demands[i] for i in range(n)) / n
        reward         = 2.0 * delivery_ratio - 1.0
        reward        += 0.1 * (self.battery / MAX_BATTERY)

        # --- 5. Advance time slot ---
        self.time_slot = (self.time_slot + 1) % 8

        # --- 6. Update loads: base demand refreshes for new slot, some join/leave ---
        for load in self.loads:
            load[0] = random.randint(3, 8)   # new base demand; type + anomaly flag persist

        if len(self.loads) > MIN_LOADS and random.random() < PROB_LOAD_REMOVED:
            self.loads.pop(random.randrange(len(self.loads)))

        if len(self.loads) < MAX_LOADS and random.random() < PROB_LOAD_ADDED:
            self.loads.append(
                [random.randint(3, 8), random.choice(ALL_TYPES), random.random() < ANOMALY_LOAD_PROB]
            )

        # --- 7. Refresh generation ---
        self.generation = random.randint(5, 15)
        next_forecast   = self._noisy_forecast(self.generation)

        done    = self._state.step_count >= MAX_STEPS
        message = (
            f"Slot {self.time_slot}: {TIME_LABELS[self.time_slot]} | "
            f"{len(self.loads)} loads"
        )
        if done:
            message = "Day complete!"

        # Pre-compute demands *once* here and pass to _make_obs so the
        # observation reflects the same values used for reward, not a
        # fresh resample (demand is stochastic).
        next_demands = [self._current_demand(i) for i in range(len(self.loads))]

        # demand_trend: did total demand rise, stay flat, or fall vs this step?
        next_total = sum(next_demands)
        if next_total > sum(demands) * 1.1:
            demand_trend = 1
        elif next_total < sum(demands) * 0.9:
            demand_trend = -1
        else:
            demand_trend = 0
        self.prev_total_demand = sum(demands)

        return self._make_obs(next_demands, reward=reward, done=done, message=message, generation_forecast=next_forecast, demand_trend=demand_trend)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _noisy_forecast(self, gen: int) -> int:
        """Return a ±20% noisy estimate of actual generation — what the agent sees."""
        return max(1, round(gen * random.uniform(0.8, 1.2)))

    def _current_demand(self, i: int) -> int:
        """Sample actual demand for load i from its (min, max) range for the current slot.

        The slot index is shifted by _week_offset (rotates which time-of-day is peak),
        so the agent cannot hard-code time-of-day → battery strategy.
        Anomalous loads draw from ANOMALY_RANGE regardless of type or slot.
        """
        base, rtype, is_anomalous = self.loads[i]
        slot       = (self.time_slot + self._week_offset) % 8
        lo, hi     = ANOMALY_RANGE if is_anomalous else DEMAND_PROFILES[rtype][slot]
        multiplier = random.uniform(lo, hi)
        return max(1, round(base * multiplier))

    def _type_weights(self, action: EnergyGridAction) -> dict:
        """Map each RegionType to its numeric weight from the action."""
        weight_map = {AllocWeight.NORMAL: 1.0, AllocWeight.HIGH: 2.0}
        return {
            RegionType.RESIDENTIAL: weight_map[action.residential],
            RegionType.INDUSTRIAL:  weight_map[action.industrial],
            RegionType.COMMERCIAL:  weight_map[action.commercial],
            RegionType.HOSPITAL:    weight_map[action.hospital],
        }

    def _priority_order(self, action: EnergyGridAction, demands: list[int]) -> list[int]:
        """Sort load indices by (weight × demand) descending — highest-priority loads get battery first."""
        tw = self._type_weights(action)
        return sorted(range(len(demands)), key=lambda i: -(tw[self.loads[i][1]] * demands[i]))

    def _distribute(self, action: EnergyGridAction, total: int, demands: list[int]) -> list[int]:
        """Allocate power proportionally to each load's (weight × demand).
        The weights come from the action, learned by the agent — no hard-coded strategy.
        """
        n  = len(demands)
        tw = self._type_weights(action)
        weighted = [demands[i] * tw[self.loads[i][1]] for i in range(n)]
        total_w  = sum(weighted)
        if total_w == 0:
            base, extra = divmod(total, n)
            return [base + (1 if i < extra else 0) for i in range(n)]
        alloc     = [int(total * w / total_w) for w in weighted]
        remainder = total - sum(alloc)
        for i in sorted(range(n), key=lambda i: -weighted[i])[:remainder]:
            alloc[i] += 1
        return alloc

    def _make_obs(self, demands: list[int], reward: float, done: bool, message: str, generation_forecast: int = 0, demand_trend: int = 0) -> EnergyGridObservation:
        types = [load[1] for load in self.loads]
        return EnergyGridObservation(
            time_slot        = self.time_slot,
            time_label       = TIME_LABELS[self.time_slot],
            num_loads        = len(self.loads),
            total_demand     = sum(demands),
            max_load         = max(demands),
            min_load         = min(demands),
            num_residential  = types.count(RegionType.RESIDENTIAL),
            num_industrial   = types.count(RegionType.INDUSTRIAL),
            num_commercial   = types.count(RegionType.COMMERCIAL),
            num_hospital     = types.count(RegionType.HOSPITAL),
            generation_forecast = generation_forecast,
            battery             = self.battery,
            demand_trend        = demand_trend,
            reward           = reward,
            done             = done,
            message          = message,
        )

