"""
Root-level server entry point for the Energy Grid environment.
This file satisfies the OpenEnv validator's requirement for server/app.py.
"""
from energy_grid_env.server.app import app


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
