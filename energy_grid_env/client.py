from __future__ import annotations

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State
except ImportError:
    from core.client_types import StepResult
    from core.env_client import EnvClient
    from core.env_server.types import State

from .models import DistributeAction, EnergyGridAction, EnergyGridObservation


class EnergyGridClient(EnvClient[EnergyGridAction, EnergyGridObservation, State]):
    """WebSocket client for the Energy Grid environment."""

    def step_action(self, action: DistributeAction) -> StepResult[EnergyGridObservation]:
        return super().step(EnergyGridAction(action=action))

    def _step_payload(self, action: EnergyGridAction) -> dict:
        return action.model_dump()

    def _parse_result(self, data: dict) -> StepResult[EnergyGridObservation]:
        return StepResult(
            observation=EnergyGridObservation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data.get("info", {}),
        )

    def _parse_state(self, data: dict) -> State:
        return State(**data)
