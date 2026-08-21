"""포트폴리오 성과·위험 지표 계산.

이 모듈은 순수 계산만 담당한다. 네트워크에 접근하지 않고, 입출력은 모두
pandas 객체 또는 스칼라다. 연율화 기준은 거래일 252일이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

def to_returns(prices: pd.DataFrame | pd.Series, log: bool = False) -> pd.DataFrame | pd.Series:
    """가격 시계열을 수익률 시계열로 변환한다.

    첫 행은 수익률을 정의할 수 없으므로 제거한다.
    """
    if log:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.iloc[1:]


def cumulative_growth(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """수익률을 누적 성장 곡선(시작값 1.0)으로 변환한다."""
    return (1.0 + returns).cumprod()


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """기하평균 기준 연환산 수익률(CAGR)."""
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    total_growth = float((1.0 + returns).prod())
    if total_growth <= 0:
        return -1.0
    return total_growth ** (periods / len(returns)) - 1.0


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """연환산 표준편차(변동성)."""
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """샤프지수 = (연환산 수익률 - 무위험수익률) / 연환산 변동성.

    ``risk_free``는 연율 기준(예: 0.035 = 3.5%)으로 받는다.
    """
    vol = annualized_volatility(returns, periods)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return (annualized_return(returns, periods) - risk_free) / vol


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """소르티노지수. 하방 변동성만을 위험으로 본다."""
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    daily_mar = (1.0 + risk_free) ** (1.0 / periods) - 1.0
    downside = returns[returns < daily_mar] - daily_mar
    if len(downside) == 0:
        return float("inf")
    downside_dev = float(np.sqrt((downside**2).sum() / len(returns)) * np.sqrt(periods))
    if downside_dev == 0:
        return float("inf")
    return (annualized_return(returns, periods) - risk_free) / downside_dev


def drawdown_series(returns: pd.Series) -> pd.Series:
    """전고점 대비 하락률 시계열(음수)."""
    equity = cumulative_growth(returns)
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """최대낙폭(MDD). 음수로 반환한다."""
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """칼마지수 = 연환산 수익률 / |최대낙폭|."""
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return annualized_return(returns, periods) / abs(mdd)


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """역사적 VaR. 하위 (1-level) 분위 일간 수익률을 음수로 반환한다."""
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    return float(np.percentile(returns, (1.0 - level) * 100.0))


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    """조건부 VaR(기대손실). VaR을 넘는 손실들의 평균."""
    returns = returns.dropna()
    if len(returns) == 0:
        return float("nan")
    threshold = value_at_risk(returns, level)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return threshold
    return float(tail.mean())


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """벤치마크 대비 베타."""
    joined = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return float("nan")
    asset, bench = joined.iloc[:, 0], joined.iloc[:, 1]
    bench_var = float(bench.var(ddof=1))
    if bench_var == 0:
        return float("nan")
    return float(asset.cov(bench) / bench_var)


def covariance_matrix(
    returns: pd.DataFrame,
    annualize: bool = True,
    shrinkage: float = 0.0,
    periods: int = TRADING_DAYS,
) -> pd.DataFrame:
    """표본 공분산 행렬.

    ``shrinkage``(0~1)를 주면 대각행렬 방향으로 축소한다. 종목 수가 관측일 수에
    비해 많을 때 표본 공분산이 불안정해지는 문제를 완화한다.
    """
    cov = returns.cov()
    if shrinkage > 0:
        target = pd.DataFrame(
            np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns
        )
        cov = (1.0 - shrinkage) * cov + shrinkage * target
    if annualize:
        cov = cov * periods
    return cov


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """상관계수 행렬."""
    return returns.corr()


def expected_returns(
    returns: pd.DataFrame, periods: int = TRADING_DAYS, method: str = "mean"
) -> pd.Series:
    """종목별 연환산 기대수익률.

    ``method``:
        - ``"mean"``  산술평균 연환산. 최적화 입력으로 표준적이다.
        - ``"cagr"``  기하평균 연환산. 실제 체감 수익률에 가깝다.
    """
    if method == "cagr":
        return returns.apply(lambda col: annualized_return(col, periods))
    if method == "mean":
        return returns.mean() * periods
    raise ValueError(f"알 수 없는 기대수익률 방식입니다: {method}")


def summarize(
    returns: pd.Series,
    risk_free: float = 0.0,
    benchmark: pd.Series | None = None,
    periods: int = TRADING_DAYS,
) -> dict[str, float]:
    """성과 지표를 한 번에 계산해 딕셔너리로 반환한다."""
    result = {
        "연환산수익률": annualized_return(returns, periods),
        "연환산변동성": annualized_volatility(returns, periods),
        "샤프지수": sharpe_ratio(returns, risk_free, periods),
        "소르티노지수": sortino_ratio(returns, risk_free, periods),
        "최대낙폭": max_drawdown(returns),
        "칼마지수": calmar_ratio(returns, periods),
        "일간VaR95": value_at_risk(returns, 0.95),
        "일간CVaR95": conditional_var(returns, 0.95),
        "누적수익률": float((1.0 + returns.dropna()).prod() - 1.0),
    }
    if benchmark is not None:
        result["베타"] = beta(returns, benchmark)
    return result
