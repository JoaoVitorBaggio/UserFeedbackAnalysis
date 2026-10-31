from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_sat_dsat.jsonl"


@pytest.fixture
def tiny_path() -> Path:
    return FIXTURE
