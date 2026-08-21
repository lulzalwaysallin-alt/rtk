"""백테스트 검증. 손익 계산이 맞는지 불변식으로 확인한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio import backtest, metrics


def test_리밸런싱_날짜는_각_주기의_첫_거래일():
    index = pd.bdate_range("2024-01-01", "2024-12-31")
    monthly = backtest.rebalance_dates(index, "monthly")
    assert len(monthly) == 12
    assert monthly[0] == index[0]
    assert {d.month for d in monthly} == set(range(1, 13))
    assert all(d in index for d in monthly), "휴장일이 아닌 실제 거래일이어야 한다"


def test_주기별_리밸런싱_횟수():
    index = pd.bdate_range("2022-01-03", "2024-12-31")
    assert len(backtest.rebalance_dates(index, "none")) == 1
    assert len(backtest.rebalance_dates(index, "yearly")) == 3
    assert len(backtest.rebalance_dates(index, "quarterly")) == 12
    assert len(backtest.rebalance_dates(index, "monthly")) == 36


def test_알_수_없는_주기는_거절():
    with pytest.raises(ValueError, match="알 수 없는 리밸런싱"):
        backtest.rebalance_dates(pd.bdate_range("2024-01-01", periods=10), "격주")


def test_단일종목_100퍼센트는_그_종목_수익률과_정확히_같다(sample_prices):
    weights = pd.Series(0.0, index=sample_prices.columns)
    weights["고성장주"] = 1.0
    result = backtest.run_fixed_weight(sample_prices, weights, "monthly", cost_bps=0)
    direct = metrics.to_returns(sample_prices["고성장주"])
    assert np.allclose(result.returns.values, direct.values, atol=1e-12)
    assert len(result.returns) == len(direct), "첫 거래일 수익률이 누락되면 안 된다"


def test_매수후보유는_단순_가중합과_같다(sample_prices):
    weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=sample_prices.columns)
    result = backtest.run_fixed_weight(sample_prices, weights, "none", cost_bps=0)
    normalized = sample_prices / sample_prices.iloc[0]
    manual = (normalized * weights).sum(axis=1)
    assert np.allclose(result.equity.values, manual.iloc[1:].values, atol=1e-10)


def test_거래비용이_클수록_성과가_낮아진다(sample_prices):
    weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=sample_prices.columns)
    free = backtest.run_fixed_weight(sample_prices, weights, "monthly", cost_bps=0)
    costly = backtest.run_fixed_weight(sample_prices, weights, "monthly", cost_bps=100)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert free.total_cost == pytest.approx(0.0)
    assert costly.total_cost > 0


def test_리밸런싱이_잦을수록_비용이_크다(sample_prices):
    weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=sample_prices.columns)
    monthly = backtest.run_fixed_weight(sample_prices, weights, "monthly", cost_bps=30)
    yearly = backtest.run_fixed_weight(sample_prices, weights, "yearly", cost_bps=30)
    assert monthly.total_cost > yearly.total_cost


def test_비중은_정규화된다(sample_prices):
    """합이 1이 아닌 비중을 넣어도 알아서 맞춰야 한다."""
    raw = pd.Series([2.0, 1.0, 1.0, 0.0], index=sample_prices.columns)
    result = backtest.run_fixed_weight(sample_prices, raw, "none", cost_bps=0)
    assert result.weights_history.iloc[0].sum() == pytest.approx(1.0)


def test_비중_합이_0이면_거절(sample_prices):
    zeros = pd.Series(0.0, index=sample_prices.columns)
    with pytest.raises(ValueError, match="비중의 합이 0"):
        backtest.run_fixed_weight(sample_prices, zeros, "none")


def test_워크포워드는_학습기간_이후부터_시작한다(sample_prices):
    result = backtest.run_walk_forward(
        sample_prices, "max_sharpe", lookback_days=252, frequency="quarterly"
    )
    returns = metrics.to_returns(sample_prices)
    assert result.returns.index[0] >= returns.index[252]
    assert not result.weights_history.empty
    assert result.weights_history.sum(axis=1).round(6).eq(1.0).all()


def test_워크포워드는_데이터가_부족하면_이유를_알려준다(sample_prices):
    with pytest.raises(ValueError, match="데이터가 짧습니다"):
        backtest.run_walk_forward(sample_prices.iloc[:100], lookback_days=252)


def test_워크포워드는_미래_정보를_쓰지_않는다(sample_prices):
    """마지막 구간 가격을 바꿔도 그 이전 비중은 그대로여야 한다."""
    base = backtest.run_walk_forward(
        sample_prices, "max_sharpe", lookback_days=252, frequency="yearly"
    )
    tampered = sample_prices.copy()
    tampered.iloc[-60:] = tampered.iloc[-60:] * 3.0  # 마지막 구간만 급등시킨다
    altered = backtest.run_walk_forward(
        tampered, "max_sharpe", lookback_days=252, frequency="yearly"
    )
    shared = base.weights_history.index.intersection(altered.weights_history.index)[:-1]
    assert len(shared) > 0
    assert np.allclose(
        base.weights_history.loc[shared].values,
        altered.weights_history.loc[shared].values,
        atol=1e-8,
    ), "미래 가격이 과거 시점의 비중에 영향을 주면 안 된다"


def test_벤치마크는_지수_수익률을_그대로_쓴다(sample_benchmark):
    result = backtest.run_benchmark(sample_benchmark, sample_benchmark.index[0], "코스피")
    assert result.name == "코스피"
    assert np.allclose(result.returns.values, metrics.to_returns(sample_benchmark).values)


def test_벤치마크_구간이_없으면_거절(sample_benchmark):
    with pytest.raises(ValueError, match="데이터가 부족"):
        backtest.run_benchmark(sample_benchmark, "2099-01-01", "코스피")


def test_비교표는_이름이_겹쳐도_행을_잃지_않는다(sample_prices):
    weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=sample_prices.columns)
    first = backtest.run_fixed_weight(sample_prices, weights, "none", 0, name="같은이름")
    second = backtest.run_fixed_weight(sample_prices, weights, "monthly", 50, name="같은이름")
    table = backtest.compare_results([first, second])
    assert len(table) == 2


def test_곡선_정렬은_공통_시작점을_1로_맞춘다(sample_prices, sample_benchmark):
    weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=sample_prices.columns)
    results = [
        backtest.run_fixed_weight(sample_prices, weights, "none", 0, name="포트폴리오"),
        backtest.run_benchmark(sample_benchmark, sample_prices.index[0], "코스피"),
    ]
    curves = backtest.align_curves(results)
    assert list(curves.columns) == ["포트폴리오", "코스피"]
    assert curves.iloc[0].tolist() == pytest.approx([1.0, 1.0])
    assert not curves.isna().any().any()


def test_요약에_거래비용과_리밸런싱_횟수가_들어간다(sample_prices):
    weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=sample_prices.columns)
    result = backtest.run_fixed_weight(sample_prices, weights, "quarterly", 20)
    summary = result.summary(0.03)
    assert summary["리밸런싱횟수"] > 1
    assert summary["총거래비용"] > 0
