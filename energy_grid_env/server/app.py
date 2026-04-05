"""
FastAPI server entry point for the Energy Grid environment.
Run with: uvicorn energy_grid_env.server.app:app --port 8000
"""
try:
    from openenv.core.env_server import create_app
except ImportError:
    from core.env_server import create_app

from ..models import EnergyGridAction, EnergyGridObservation
from .energy_grid_environment import EnergyGridEnvironment

app = create_app(
    EnergyGridEnvironment,
    EnergyGridAction,
    EnergyGridObservation,
    env_name="energy_grid",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
