"""
Root-level server entry point — re-exports the Energy Grid FastAPI app.
This file satisfies the OpenEnv validator's requirement for server/app.py.
"""
from energy_grid_env.server.app import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
