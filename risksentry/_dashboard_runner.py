from __future__ import annotations

"""Convenience entry-point: ``risksentry-dashboard``."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).parent.parent / "frontend" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", "8501"]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
