"""테스트 공통 설정과 픽스처."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """테스트가 사용자의 실제 캐시 디렉터리를 건드리지 않게 한다."""
    from portfolio import data

    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    yield


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """4종목 · 약 3.5년치 가격표.

    기하 브라운 운동으로 만들되 시드를 고정해, 같은 입력이면 항상 같은 결과가
    나오도록 한다. 종목마다 추세·변동성·상관을 다르게 주어 최적화가 의미 있는
    선택을 하도록 설계했다.
    """
    index = pd.bdate_range("2021-01-04", periods=900)
    rng = np.random.default_rng(42)
    market = rng.normal(0.0004, 0.010, len(index))  # 공통 시장 요인

    specs = {
        "고성장주": (0.0007, 0.016, 1.3),
        "안정주": (0.0002, 0.007, 0.5),
        "해외주": (0.0005, 0.013, 0.9),
        "채권ETF": (0.0001, 0.005, -0.2),
    }
    columns = {}
    for name, (drift, vol, beta) in specs.items():
        idiosyncratic = rng.normal(0.0, vol, len(index))
        daily = drift + beta * market + idiosyncratic
        columns[name] = 100.0 * np.exp(np.cumsum(daily))
    return pd.DataFrame(columns, index=index)


@pytest.fixture
def sample_benchmark() -> pd.Series:
    index = pd.bdate_range("2021-01-04", periods=900)
    rng = np.random.default_rng(99)
    return pd.Series(
        2500.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(index)))),
        index=index,
        name="코스피",
    )


@pytest.fixture
def two_asset_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """해석적으로 정답을 알 수 있는 2종목 입력.

    무상관·동일분산이므로 최소분산은 50:50, 최대샤프(rf=0)는 mu 에 비례한다.
    """
    names = ["A", "B"]
    mu = pd.Series([0.10, 0.20], index=names)
    cov = pd.DataFrame(np.diag([0.04, 0.04]), index=names, columns=names)
    return mu, cov
