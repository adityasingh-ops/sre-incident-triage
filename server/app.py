"""
OpenEnv-compatible entry point for the SRE Incident Triage environment.
This file provides the main() function required by openenv_serve.
"""
import uvicorn
from server.main import app


def main():
    """Main entry point for openenv serve"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
