"""리밸런싱 백테스트.

두 가지 방식을 제공한다.

``고정비중``
    지금 계산한 최적 비중으로 과거에 투자했다면 어땠을지 본다. 이해하기 쉽지만
    "지금 알고 있는 정답"을 과거에 적용하는 것이라 실제보다 좋게 나온다.

``워크포워드``
    매 리밸런싱 시점에 그때까지의 데이터만으로 다시 최적화한다. 미래 정보를
    쓰지 않으므로 실제 운용에 훨씬 가깝다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics
from .optimize import optimize

REBALANCE_FREQUENCIES: dict[str, str] = {
    "none": "리밸런싱 없음 (매수 후 보유)",
    "monthly": "매월",
    "quarterly": "분기별",
    "yearly": "매년",
}


@dataclass
class BacktestResult:
    """백테스트 한 건의 결과."""

    name: str
    returns: pd.Series
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    total_cost: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def equity(self) -> pd.Series:
        """1.0 에서 시작하는 누적 성장 곡선."""
        curve = metrics.cumulative_growth(self.returns)
        curve.name = self.name
        return curve

    def summary(self, risk_free: float = 0.0, benchmark: pd.Series | None = None) -> dict:
        stats = metrics.summarize(self.returns, risk_free, benchmark)
        stats["총거래비용"] = self.total_cost
        stats["리밸런싱횟수"] = int(len(self.weights_history))
        return stats


def rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """리밸런싱을 실행할 실제 거래일들을 고른다.

    달력상의 1일이 휴장일 수 있으므로, 각 주기의 '첫 거래일'을 쓴다.
    """
    if frequency not in REBALANCE_FREQUENCIES:
        raise ValueError(f"알 수 없는 리밸런싱 주기입니다: {frequency}")
    if len(index) == 0:
        return pd.DatetimeIndex([])
    if frequency == "none":
        return pd.DatetimeIndex([index[0]])

    period = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}[frequency]
    as_series = pd.Series(index, index=index)
    firsts = as_series.groupby(index.to_period(period)).first()
    return pd.DatetimeIndex(firsts.values).sort_values()


def _simulate(
    returns: pd.DataFrame,
    weight_schedule: dict[pd.Timestamp, pd.Series],
    cost_bps: float,
    name: str,
) -> BacktestResult:
    """비중 일정표를 받아 일간 손익을 굴린다.

    Args:
        returns: 종목별 일간 수익률.
        weight_schedule: 리밸런싱 날짜 → 그날 맞출 목표 비중.
        cost_bps: 편도 거래비용(bp). 회전율에 곱해 차감한다.
    """
    columns = list(returns.columns)
    schedule = {pd.Timestamp(k): v.reindex(columns).fillna(0.0) for k, v in weight_schedule.items()}
    if not schedule:
        raise ValueError("리밸런싱 일정이 비어 있습니다.")

    first_date = min(schedule)
    active = returns.loc[returns.index >= first_date]
    if active.empty:
        raise ValueError("백테스트할 구간이 없습니다. 기간을 늘려주세요.")

    # 첫 비중은 첫 수익률이 발생하기 직전(전일 종가)에 맞춘 것으로 본다.
    # 그래야 조회한 첫 거래일의 수익률까지 빠짐없이 반영된다.
    values = schedule[first_date].to_numpy(dtype=float).copy()  # 시작 자산 1.0 기준
    daily_returns: list[float] = []
    dates: list[pd.Timestamp] = []
    turnovers: dict[pd.Timestamp, float] = {}
    recorded_weights: dict[pd.Timestamp, pd.Series] = {first_date: schedule[first_date]}
    total_cost = 0.0

    for timestamp, row in active.iterrows():
        before = values.sum()
        values = values * (1.0 + row.to_numpy(dtype=float))
        after_drift = values.sum()

        # 최초 매수일에는 이미 목표 비중이므로 다시 맞출 것이 없다.
        if timestamp in schedule and timestamp != first_date:
            target = schedule[timestamp].to_numpy(dtype=float) * after_drift
            turnover = float(np.abs(target - values).sum() / after_drift) if after_drift > 0 else 0.0
            cost = turnover * cost_bps / 10_000.0
            values = target * (1.0 - cost)
            total_cost += cost * after_drift
            turnovers[timestamp] = turnover
            recorded_weights[timestamp] = schedule[timestamp]

        dates.append(timestamp)
        daily_returns.append(values.sum() / before - 1.0 if before > 0 else 0.0)

    series = pd.Series(daily_returns, index=pd.DatetimeIndex(dates), name=name)
    return BacktestResult(
        name=name,
        returns=series,
        weights_history=pd.DataFrame(recorded_weights).T.sort_index(),
        turnover=pd.Series(turnovers, dtype=float).sort_index(),
        total_cost=total_cost,
    )


def run_fixed_weight(
    prices: pd.DataFrame,
    weights: pd.Series,
    frequency: str = "quarterly",
    cost_bps: float = 15.0,
    name: str = "최적 포트폴리오",
) -> BacktestResult:
    """정해진 비중을 주기적으로 다시 맞추며 굴린다."""
    returns = metrics.to_returns(prices)
    weights = weights.reindex(prices.columns).fillna(0.0)
    if weights.sum() <= 0:
        raise ValueError("비중의 합이 0입니다.")
    weights = weights / weights.sum()

    schedule = {date: weights for date in rebalance_dates(returns.index, frequency)}
    result = _simulate(returns, schedule, cost_bps, name)
    if frequency == "none":
        result.notes.append("매수 후 보유 기준입니다. 시간이 지나면 비중이 저절로 벌어집니다.")
    return result


def run_walk_forward(
    prices: pd.DataFrame,
    method: str = "max_sharpe",
    lookback_days: int = 252,
    frequency: str = "quarterly",
    cost_bps: float = 15.0,
    risk_free: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    shrinkage: float = 0.1,
    name: str = "워크포워드",
) -> BacktestResult:
    """매 리밸런싱 시점에 과거 데이터만으로 다시 최적화하며 굴린다.

    미래 정보를 쓰지 않는다는 점에서 고정비중 백테스트보다 보수적이고 현실적이다.
    """
    returns = metrics.to_returns(prices)
    if len(returns) <= lookback_days:
        raise ValueError(
            f"학습기간 {lookback_days}일보다 데이터가 짧습니다"
            f"(현재 {len(returns)}일). 조회 기간을 늘리거나 학습기간을 줄여주세요."
        )

    candidates = rebalance_dates(returns.index, frequency)
    positions = returns.index.get_indexer(candidates)
    usable = [(d, p) for d, p in zip(candidates, positions, strict=True) if p >= lookback_days]
    if not usable:
        raise ValueError(
            "학습기간을 확보한 리밸런싱 시점이 없습니다. 조회 기간을 늘려주세요."
        )

    schedule: dict[pd.Timestamp, pd.Series] = {}
    notes: list[str] = []
    equal = pd.Series(1.0 / len(prices.columns), index=prices.columns)
    for rebal_date, position in usable:
        window = returns.iloc[position - lookback_days : position]
        try:
            mu = metrics.expected_returns(window)
            cov = metrics.covariance_matrix(window, shrinkage=shrinkage)
            outcome = optimize(
                mu, cov, method, risk_free=risk_free,
                min_weight=min_weight, max_weight=max_weight,
            )
            schedule[rebal_date] = outcome.weights
        except Exception as exc:
            schedule[rebal_date] = equal
            notes.append(f"{rebal_date:%Y-%m-%d}: 최적화 실패로 동일비중 사용 ({exc})")

    result = _simulate(returns, schedule, cost_bps, name)
    result.notes.extend(notes)
    return result


def run_benchmark(benchmark_prices: pd.Series, start, name: str) -> BacktestResult:
    """벤치마크 지수를 같은 구간으로 잘라 비교 대상으로 만든다."""
    series = benchmark_prices.loc[benchmark_prices.index >= pd.Timestamp(start)]
    if len(series) < 2:
        raise ValueError(f"'{name}' 비교 구간의 데이터가 부족합니다.")
    returns = metrics.to_returns(series)
    returns.name = name
    return BacktestResult(name=name, returns=returns)


def compare_results(
    results: list[BacktestResult], risk_free: float = 0.0
) -> pd.DataFrame:
    """여러 백테스트 결과를 나란히 비교하는 표로 만든다."""
    rows: dict[str, dict] = {}
    for result in results:
        if result.returns.empty:
            continue
        # 이름이 겹치면 앞의 결과가 조용히 사라지므로 번호를 붙여 구분한다.
        label, suffix = result.name, 2
        while label in rows:
            label, suffix = f'{result.name} ({suffix})', suffix + 1
        rows[label] = result.summary(risk_free)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T


def align_curves(results: list[BacktestResult]) -> pd.DataFrame:
    """비교용 누적수익 곡선들을 공통 시작일에 맞춰 1.0 으로 재정규화한다."""
    curves = [r.equity for r in results if not r.returns.empty]
    if not curves:
        return pd.DataFrame()
    frame = pd.concat(curves, axis=1).dropna(how="all").ffill().dropna(how="any")
    if frame.empty:
        return frame
    return frame / frame.iloc[0]
