"""지표 계산 검증. 손으로 답을 낼 수 있는 입력으로만 확인한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio import metrics


def test_수익률_변환은_첫행을_버린다():
    prices = pd.Series([100.0, 110.0, 99.0])
    returns = metrics.to_returns(prices)
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)


def test_누적성장은_가격비율과_같다():
    prices = pd.Series([100.0, 110.0, 121.0])
    growth = metrics.cumulative_growth(metrics.to_returns(prices))
    assert growth.iloc[-1] == pytest.approx(1.21)


def test_연환산수익률은_기하평균이다():
    # 252거래일 동안 매일 정확히 같은 수익률이면 CAGR 은 그 복리값과 일치한다
    daily = 0.0005
    returns = pd.Series([daily] * 252)
    assert metrics.annualized_return(returns) == pytest.approx((1 + daily) ** 252 - 1)


def test_연환산변동성은_루트252배():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.01, 5000))
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert metrics.annualized_volatility(returns) == pytest.approx(expected)


def test_샤프지수_정의():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0006, 0.01, 2520))
    manual = (metrics.annualized_return(returns) - 0.03) / metrics.annualized_volatility(returns)
    assert metrics.sharpe_ratio(returns, 0.03) == pytest.approx(manual)


def test_최대낙폭은_음수이고_알려진_값과_일치():
    # 100 → 120 → 60 → 90 : 고점 120 대비 60 이므로 -50%
    prices = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert metrics.max_drawdown(metrics.to_returns(prices)) == pytest.approx(-0.5)


def test_상승만_하면_낙폭이_없다():
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])
    assert metrics.max_drawdown(metrics.to_returns(prices)) == pytest.approx(0.0)


def test_소르티노는_하락이_없으면_무한대():
    returns = pd.Series([0.01] * 100)
    assert metrics.sortino_ratio(returns) == float("inf")


def test_VaR과_CVaR의_대소관계():
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0.0, 0.02, 5000))
    var = metrics.value_at_risk(returns, 0.95)
    cvar = metrics.conditional_var(returns, 0.95)
    assert cvar <= var < 0, "CVaR(꼬리 평균)은 VaR보다 나쁘거나 같아야 한다"


def test_자기자신에_대한_베타는_1():
    rng = np.random.default_rng(6)
    series = pd.Series(rng.normal(0, 0.01, 500))
    assert metrics.beta(series, series) == pytest.approx(1.0)


def test_공분산_축소는_비대각_성분만_줄인다():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=list("ABC"))
    plain = metrics.covariance_matrix(returns, shrinkage=0.0)
    shrunk = metrics.covariance_matrix(returns, shrinkage=0.5)
    assert np.allclose(np.diag(plain), np.diag(shrunk)), "대각(분산)은 유지되어야 한다"
    off_plain = plain.values[0, 1]
    off_shrunk = shrunk.values[0, 1]
    assert abs(off_shrunk) == pytest.approx(abs(off_plain) * 0.5)


def test_기대수익률_두_방식_모두_동작():
    rng = np.random.default_rng(8)
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, (1000, 2)), columns=["A", "B"])
    assert len(metrics.expected_returns(returns, method="mean")) == 2
    assert len(metrics.expected_returns(returns, method="cagr")) == 2
    with pytest.raises(ValueError, match="알 수 없는"):
        metrics.expected_returns(returns, method="없는방식")


def test_빈_입력에도_예외를_던지지_않는다():
    empty = pd.Series(dtype=float)
    assert np.isnan(metrics.annualized_return(empty))
    assert np.isnan(metrics.annualized_volatility(empty))
    assert np.isnan(metrics.max_drawdown(empty))


def test_요약은_모든_항목을_담는다(sample_prices):
    returns = metrics.to_returns(sample_prices)["고성장주"]
    summary = metrics.summarize(returns, 0.03)
    for key in ["연환산수익률", "연환산변동성", "샤프지수", "최대낙폭", "누적수익률"]:
        assert key in summary and np.isfinite(summary[key])
