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


# The 4 distribution strategies the agent can choose
class DistributeAction(str, Enum):
    EQUAL        = "EQUAL"        # Split generation evenly across all active loads
    MIN_FIRST    = "MIN_FIRST"    # Satisfy smallest loads first (maximise fully-met count)
    MAX_FIRST    = "MAX_FIRST"    # Satisfy largest loads first (protect critical loads)
    PROPORTIONAL = "PROPORTIONAL" # Allocate proportionally to each load's demand


class EnergyGridAction(Action):
    action: DistributeAction = Field(..., description="Distribution strategy")


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
    generation:   int   = Field(..., description="Power being generated this step")
    battery:      int   = Field(..., description="Energy stored in battery")

    reward:       float = Field(0.0,   description="Reward from the last action")
    done:         bool  = Field(False, description="Whether the episode has ended")
    message:      str   = Field("",    description="Human-readable status message")
