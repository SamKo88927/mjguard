from __future__ import annotations

from pathlib import Path

import pytest

from mjguard.config import RobotConfig, load
from mjguard.executor import Executor, SimTransport

HERE = Path(__file__).parent


@pytest.fixture
def config() -> RobotConfig:
    return load(HERE / "three_link.toml")


@pytest.fixture
def executor(config: RobotConfig) -> Executor:
    return Executor(config, SimTransport(config))
