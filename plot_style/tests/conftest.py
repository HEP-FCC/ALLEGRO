import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


@pytest.fixture(scope="session")
def output_dir():
    """Where example plots are written for visual inspection (kept after the tests)."""
    d = Path(os.environ.get("ALLEGRO_STYLE_OUTPUT", HERE / "output"))
    d.mkdir(parents=True, exist_ok=True)
    return d
