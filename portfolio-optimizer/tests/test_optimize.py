"""최적화 검증.

가능한 곳에서는 해석적으로 정답을 아는 입력을 써서, 최적화기가 '수렴했다'가
아니라 '정답을 냈다'를 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio import metrics
from portfolio import optimize as opt


def test_최소분산은_무상관_동일분산에서_반반(two_asset_inputs):
    mu, cov = two_asset_inputs
    result = opt.optimize(mu, cov, "min_variance")
    assert result.weights.tolist() == pytest.approx([0.5, 0.5], abs=1e-4)


def test_최대샤프는_무상관에서_mu에_비례(two_asset_inputs):
    # rf=0, 무상관, 동일분산이면 w_i ∝ mu_i/var_i → 1:2
    mu, cov = two_asset_inputs
    result = opt.optimize(mu, cov, "max_sharpe", risk_free=0.0)
    assert result.weights.tolist() == pytest.approx([1 / 3, 2 / 3], abs=1e-4)


def test_위험균등은_변동성에_반비례():
    names = ["저변동", "고변동"]
    mu = pd.Series([0.08, 0.15], index=names)
    cov = pd.DataFrame(np.diag([0.04, 0.16]), index=names, columns=names)  # 변동성 20% vs 40%
    result = opt.optimize(mu, cov, "risk_parity")
    assert result.weights.tolist() == pytest.approx([2 / 3, 1 / 3], abs=1e-3)


def test_위험균등은_위험기여도를_실제로_균등하게_만든다(sample_prices):
    returns = metrics.to_returns(sample_prices)
    mu = metrics.expected_returns(returns)
    cov = metrics.covariance_matrix(returns)
    result = opt.optimize(mu, cov, "risk_parity")
    contributions = opt.risk_contributions(result.weights.to_numpy(), cov.to_numpy())
    assert contributions == pytest.approx([0.25] * 4, abs=5e-3)


def test_동일비중은_정확히_1_over_n(sample_prices):
    returns = metrics.to_returns(sample_prices)
    result = opt.optimize(
        metrics.expected_returns(returns), metrics.covariance_matrix(returns), "equal_weight"
    )
    assert result.weights.tolist() == pytest.approx([0.25] * 4)


def test_목표수익률은_정확히_달성된다(two_asset_inputs):
    mu, cov = two_asset_inputs
    result = opt.optimize(mu, cov, "target_return", target_return=0.15)
    assert result.expected_return == pytest.approx(0.15, abs=1e-6)


def test_달성_불가능한_목표수익률은_안내와_함께_거절(two_asset_inputs):
    mu, cov = two_asset_inputs  # 최대 20%
    with pytest.raises(ValueError, match="달성할 수 없습니다"):
        opt.optimize(mu, cov, "target_return", target_return=0.50)


def test_목표수익률에_값이_없으면_거절(two_asset_inputs):
    mu, cov = two_asset_inputs
    with pytest.raises(ValueError, match="target_return"):
        opt.optimize(mu, cov, "target_return")


@pytest.mark.parametrize("method", list(opt.STRATEGIES))
def test_모든_전략에서_비중_합은_1이고_음수가_없다(sample_prices, method):
    returns = metrics.to_returns(sample_prices)
    mu, cov = metrics.expected_returns(returns), metrics.covariance_matrix(returns)
    kwargs = {"target_return": float(mu.mean())} if method == "target_return" else {}
    result = opt.optimize(mu, cov, method, **kwargs)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (result.weights >= -1e-9).all(), "롱온리 제약이 지켜져야 한다"


def test_비중_상한이_지켜진다(sample_prices):
    returns = metrics.to_returns(sample_prices)
    result = opt.optimize(
        metrics.expected_returns(returns), metrics.covariance_matrix(returns),
        "max_sharpe", max_weight=0.30,
    )
    assert result.weights.max() <= 0.30 + 1e-6


def test_비중_하한이_지켜진다(sample_prices):
    returns = metrics.to_returns(sample_prices)
    result = opt.optimize(
        metrics.expected_returns(returns), metrics.covariance_matrix(returns),
        "max_sharpe", min_weight=0.15,
    )
    assert result.weights.min() >= 0.15 - 1e-6


def test_불가능한_상한은_계산법까지_알려주며_거절(two_asset_inputs):
    mu, cov = two_asset_inputs  # 2종목에 상한 30% → 최대 60%
    with pytest.raises(ValueError, match="100%를 만들 수 없습니다"):
        opt.optimize(mu, cov, "max_sharpe", max_weight=0.30)


def test_불가능한_하한도_거절(two_asset_inputs):
    mu, cov = two_asset_inputs  # 2종목에 하한 60% → 최소 120%
    with pytest.raises(ValueError, match="100%를 넘습니다"):
        opt.optimize(mu, cov, "max_sharpe", min_weight=0.60)


def test_최대샤프는_다른_전략보다_샤프가_높다(sample_prices):
    returns = metrics.to_returns(sample_prices)
    mu, cov = metrics.expected_returns(returns), metrics.covariance_matrix(returns)
    best = opt.optimize(mu, cov, "max_sharpe", risk_free=0.02)
    for method in ["min_variance", "risk_parity", "equal_weight"]:
        other = opt.optimize(mu, cov, method, risk_free=0.02)
        assert best.sharpe >= other.sharpe - 1e-6, f"{method}보다 샤프가 낮으면 안 된다"


def test_최소분산은_다른_전략보다_변동성이_낮다(sample_prices):
    returns = metrics.to_returns(sample_prices)
    mu, cov = metrics.expected_returns(returns), metrics.covariance_matrix(returns)
    best = opt.optimize(mu, cov, "min_variance")
    for method in ["max_sharpe", "risk_parity", "equal_weight"]:
        other = opt.optimize(mu, cov, method)
        assert best.volatility <= other.volatility + 1e-6


def test_결과는_재현_가능하다(sample_prices):
    returns = metrics.to_returns(sample_prices)
    mu, cov = metrics.expected_returns(returns), metrics.covariance_matrix(returns)
    first = opt.optimize(mu, cov, "max_sharpe").weights
    second = opt.optimize(mu, cov, "max_sharpe").weights
    assert first.tolist() == pytest.approx(second.tolist(), abs=1e-12)


def test_종목이_하나면_전액_투자하고_안내한다():
    mu = pd.Series([0.1], index=["A"])
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    result = opt.optimize(mu, cov, "max_sharpe")
    assert result.weights.tolist() == [1.0]
    assert "분산 효과가 없습니다" in result.message


def test_알_수_없는_전략은_거절(two_asset_inputs):
    mu, cov = two_asset_inputs
    with pytest.raises(ValueError, match="알 수 없는 전략"):
        opt.optimize(mu, cov, "없는전략")


def test_순서가_어긋난_공분산은_거절(two_asset_inputs):
    mu, cov = two_asset_inputs
    with pytest.raises(ValueError, match="순서가 일치"):
        opt.optimize(mu, cov.iloc[::-1], "max_sharpe")


def test_효율적투자선은_수익률과_변동성이_함께_증가한다(sample_prices):
    returns = metrics.to_returns(sample_prices)
    frontier = opt.efficient_frontier(
        metrics.expected_returns(returns), metrics.covariance_matrix(returns), n_points=20
    )
    assert len(frontier) >= 10
    assert (frontier["변동성"].diff().dropna() >= -1e-6).all()
    assert (frontier["수익률"].diff().dropna() >= -1e-6).all()


def test_효율적투자선_위에_더_나은_점은_없다(sample_prices):
    """최대샤프 포트폴리오는 투자선 위(또는 선상)에 있어야 한다."""
    returns = metrics.to_returns(sample_prices)
    mu, cov = metrics.expected_returns(returns), metrics.covariance_matrix(returns)
    frontier = opt.efficient_frontier(mu, cov, n_points=40)
    best = opt.optimize(mu, cov, "max_sharpe", risk_free=0.02)
    nearby = frontier.iloc[(frontier["변동성"] - best.volatility).abs().argsort()[:1]]
    assert float(nearby["수익률"].iloc[0]) <= best.expected_return + 5e-3


def test_이산배분은_예산을_넘지_않는다():
    weights = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    prices = pd.Series([70000.0, 33333.0, 1250.0], index=["A", "B", "C"])
    plan = opt.discrete_allocation(weights, prices, 10_000_000)
    assert plan["매수금액"].sum() <= 10_000_000
    assert plan.attrs["잔여현금"] >= 0
    assert (plan["매수수량(주)"] >= 0).all()
    assert plan["매수수량(주)"].dtype.kind == "i", "주식 수량은 정수여야 한다"


def test_이산배분의_실제비중은_목표에_가깝다():
    weights = pd.Series([0.6, 0.4], index=["A", "B"])
    prices = pd.Series([100.0, 50.0], index=["A", "B"])
    plan = opt.discrete_allocation(weights, prices, 100_000_000)
    assert plan["실제비중(%)"].tolist() == pytest.approx([60.0, 40.0], abs=0.1)


def test_예산이_최저가보다_적으면_아무것도_못_산다():
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    prices = pd.Series([100000.0, 200000.0], index=["A", "B"])
    plan = opt.discrete_allocation(weights, prices, 1000)
    assert plan["매수수량(주)"].sum() == 0
    assert plan.attrs["잔여현금"] == pytest.approx(1000)


def test_가격이_0이면_거절():
    weights = pd.Series([1.0], index=["A"])
    with pytest.raises(ValueError, match="가격이 0 이하"):
        opt.discrete_allocation(weights, pd.Series([0.0], index=["A"]), 1000)


def test_겹치는_종목이_없으면_거절():
    with pytest.raises(ValueError, match="겹치는 종목이 없습니다"):
        opt.discrete_allocation(
            pd.Series([1.0], index=["A"]), pd.Series([100.0], index=["Z"]), 1000
        )
