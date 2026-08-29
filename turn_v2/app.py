"""Compatibility shim — the canonical demo app lives in `app/app.py`.

Keeps the documented commands working:
  uv run python turn_v2/app.py --selftest
  uv run python turn_v2/app.py
"""
import importlib.util
from pathlib import Path

_APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def _load():
    spec = importlib.util.spec_from_file_location("hinglish_turn_app", _APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    _load().main()
