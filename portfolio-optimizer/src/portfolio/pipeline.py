"""분석 오케스트레이션.

화면(Streamlit)과 계산을 분리하기 위한 계층이다. 여기서는 이미 받아온 가격표를
입력으로 받아 최적화 → 백테스트 → 지표 계산까지 한 번에 수행한다. 네트워크에
접근하지 않으므로 테스트에서 그대로 호출할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import backtest, metrics
from .optimize import (
    OptimizationResult,
    discrete_allocation,
    efficient_frontier,
    optimize,
)


@dataclass
class AnalysisConfig:
    """사용자가 화면에서 고른 설정."""

    method: str = "max_sharpe"
    risk_free: float = 0.03
    min_weight: float = 0.0
    max_weight: float = 0.40
    target_return: float | None = None
    rebalance: str = "quarterly"
    cost_bps: float = 15.0
    shrinkage: float = 0.10
    lookback_days: int = 252
    budget: float | None = None
    walk_forward: bool = True


@dataclass
class AnalysisResult:
    """한 번의 분석이 만들어낸 모든 산출물."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    expected: pd.Series
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    optimal: OptimizationResult
    equal_weight: OptimizationResult
    frontier: pd.DataFrame
    asset_points: pd.DataFrame
    backtests: list[backtest.BacktestResult]
    comparison: pd.DataFrame
    purchase_plan: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def main_backtest(self) -> backtest.BacktestResult:
        return self.backtests[0]


def analyze(
    prices: pd.DataFrame,
    config: AnalysisConfig,
    benchmarks: dict[str, pd.Series] | None = None,
) -> AnalysisResult:
    """가격표 하나로 전체 분석을 수행한다.

    Args:
        prices: 컬럼이 종목, 인덱스가 날짜인 종가표. 결측이 없어야 한다.
        config: 최적화·백테스트 설정.
        benchmarks: 이름 → 지수 종가 시계열. 비교 곡선으로 함께 그린다.
    """
    if prices.shape[1] == 0:
        raise ValueError("종목을 한 개 이상 선택해주세요.")
    if len(prices) < 30:
        raise ValueError(
            f"거래일이 {len(prices)}일뿐이라 통계가 의미 없습니다. 기간을 최소 3개월 이상으로 늘려주세요."
        )

    warnings: list[str] = []
    if len(prices) < 120:
        warnings.append(
            f"분석 구간이 {len(prices)}거래일로 짧습니다. 결과가 특정 시기에 좌우될 수 있으니 1년 이상을 권합니다."
        )
    if prices.shape[1] == 1:
        warnings.append("종목이 하나뿐이라 분산 효과를 계산할 수 없습니다. 2개 이상 담아주세요.")

    returns = metrics.to_returns(prices)
    expected = metrics.expected_returns(returns)
    covariance = metrics.covariance_matrix(returns, shrinkage=config.shrinkage)
    correlation = metrics.correlation_matrix(returns)

    optimal = optimize(
        expected, covariance, config.method,
        risk_free=config.risk_free,
        min_weight=config.min_weight,
        max_weight=config.max_weight,
        target_return=config.target_return,
    )
    if optimal.message:
        warnings.append(optimal.message)

    equal = optimize(expected, covariance, "equal_weight", risk_free=config.risk_free)

    frontier = efficient_frontier(
        expected, covariance,
        min_weight=config.min_weight, max_weight=config.max_weight,
    )
    asset_points = pd.DataFrame(
        {
            "수익률": expected,
            "변동성": pd.Series(
                {name: float(covariance.loc[name, name] ** 0.5) for name in expected.index}
            ),
        }
    )

    results = [
        backtest.run_fixed_weight(
            prices, optimal.weights, config.rebalance, config.cost_bps,
            name="추천 포트폴리오",
        ),
        backtest.run_fixed_weight(
            prices, equal.weights, config.rebalance, config.cost_bps,
            name="동일비중 (비교용)",
        ),
    ]

    if config.walk_forward and prices.shape[1] > 1:
        try:
            results.append(
                backtest.run_walk_forward(
                    prices, config.method,
                    lookback_days=config.lookback_days,
                    frequency=config.rebalance,
                    cost_bps=config.cost_bps,
                    risk_free=config.risk_free,
                    min_weight=config.min_weight,
                    max_weight=config.max_weight,
                    shrinkage=config.shrinkage,
                    name="워크포워드 (미래정보 배제)",
                )
            )
        except ValueError as exc:
            warnings.append(f"워크포워드 검증을 건너뛰었습니다. {exc}")

    for name, series in (benchmarks or {}).items():
        try:
            results.append(backtest.run_benchmark(series, prices.index[0], name))
        except Exception as exc:
            warnings.append(f"'{name}' 비교 곡선을 만들지 못했습니다. {exc}")

    comparison = backtest.compare_results(results, config.risk_free)

    purchase_plan = None
    if config.budget and config.budget > 0:
        try:
            purchase_plan = discrete_allocation(
                optimal.weights, prices.iloc[-1], config.budget
            )
        except Exception as exc:
            warnings.append(f"매수 수량을 계산하지 못했습니다. {exc}")

    return AnalysisResult(
        prices=prices,
        returns=returns,
        expected=expected,
        covariance=covariance,
        correlation=correlation,
        optimal=optimal,
        equal_weight=equal,
        frontier=frontier,
        asset_points=asset_points,
        backtests=results,
        comparison=comparison,
        purchase_plan=purchase_plan,
        warnings=warnings,
    )


def headline_metrics(result: AnalysisResult) -> dict[str, float]:
    """화면 상단 카드에 띄울 핵심 숫자 네 개."""
    realized = result.main_backtest.summary(0.0)
    return {
        "기대수익률": result.optimal.expected_return,
        "예상변동성": result.optimal.volatility,
        "샤프지수": result.optimal.sharpe,
        "과거최대낙폭": realized["최대낙폭"],
    }
