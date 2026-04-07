from __future__ import annotations

from enum import Enum
from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from core.env_server.types import Action, Observation


# Region types — each has a different demand profile over the day
class RegionType(str, Enum):
    RESIDENTIAL = "RESIDENTIAL"  # low morning, high evening (homes, apartments)
    INDUSTRIAL  = "INDUSTRIAL"   # high during work hours, low at night (factories, plants)
    COMMERCIAL  = "COMMERCIAL"   # moderate business hours, low otherwise (offices, shops)
    HOSPITAL    = "HOSPITAL"     # near-constant high demand, critical (never drops much)


class DistributeAction(str, Enum):
    EQUAL        = "EQUAL"        # Split generation evenly across all loads
    MIN_FIRST    = "MIN_FIRST"    # Satisfy smallest loads first (maximise fully-met count)
    MAX_FIRST    = "MAX_FIRST"    # Satisfy largest loads first (protect critical loads)
    PROPORTIONAL = "PROPORTIONAL" # Allocate proportionally to each load's demand


class BatteryMode(str, Enum):
    SAVE     = "SAVE"      # Don't draw battery this step — reserve for later
    MODERATE = "MODERATE"  # Draw up to half of remaining battery to cover gaps
    SPEND    = "SPEND"     # Draw as much battery as needed to cover all gaps


class EnergyGridAction(Action):
    action:       DistributeAction = Field(DistributeAction.PROPORTIONAL, description="How to distribute generated power across loads")
    battery_mode: BatteryMode      = Field(BatteryMode.SPEND,             description="How aggressively to draw battery this step")


# What the agent sees after each step
class EnergyGridObservation(Observation):
    # Time information — the key scheduling signal
    time_slot:    int   = Field(..., description="Current 3-hour time slot (0=00:00, 7=21:00)")
    time_label:   str   = Field(..., description="Human-readable time window, e.g. '06:00-09:00'")

    # Grid summary stats (reflect time-adjusted demand)
    num_loads:    int   = Field(..., description="Number of active loads this step")
    total_demand: int   = Field(..., description="Total energy demand across all loads")
    max_load:     int   = Field(..., description="Highest single load demand this slot")
    min_load:     int   = Field(..., description="Lowest single load demand this slot")

    # Region type breakdown — so agent knows the grid composition
    num_residential: int = Field(0, description="Number of residential loads")
    num_industrial:  int = Field(0, description="Number of industrial loads")
    num_commercial:  int = Field(0, description="Number of commercial loads")
    num_hospital:    int = Field(0, description="Number of hospital/critical loads")

    # Power supply
    generation_forecast: int   = Field(..., description="Forecasted power generation this step (±20% of actual)")
    battery:             int   = Field(..., description="Energy stored in battery")

    # Demand momentum — did total demand go up, stay flat, or fall vs the previous step?
    # -1 = falling  (demand dropped >10%)   → consider conserving battery
    #  0 = flat      (within ±10%)
    # +1 = rising   (demand rose >10%)      → peak may be approaching, plan ahead
    demand_trend: int = Field(0, description="Demand trend vs previous step: -1 falling, 0 flat, +1 rising")

    reward:       float = Field(0.0,   description="Reward from the last action")
    done:         bool  = Field(False, description="Whether the episode has ended")
    message:      str   = Field("",    description="Human-readable status message")
