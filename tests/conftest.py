"""pytest conftest: make the repo root importable so tests can do
``from pose_update import …`` regardless of where they live.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
