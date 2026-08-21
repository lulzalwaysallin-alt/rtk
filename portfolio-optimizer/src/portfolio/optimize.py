"""마코위츠 평균-분산 최적화와 그 변형들.

``scipy.optimize`` 의 SLSQP 로 제약 최적화를 푼다. 모든 전략은 롱온리이며
비중의 합은 1이다. 최적화가 수렴하지 않으면 예외를 던지지 않고 동일비중으로
물러선 뒤 그 사실을 ``OptimizationResult.message`` 에 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

#: 사용자에게 보여줄 전략 이름과 설명
STRATEGIES: dict[str, dict[str, str]] = {
    "max_sharpe": {
        "label": "최대 샤프지수",
        "description": "감수한 위험 한 단위당 수익이 가장 큰 조합입니다. 가장 널리 쓰이는 기준입니다.",
    },
    "min_variance": {
        "label": "최소 변동성",
        "description": "수익보다 안정성이 우선일 때. 가격 등락 폭이 가장 작은 조합을 찾습니다.",
    },
    "risk_parity": {
        "label": "위험균등 배분",
        "description": "각 종목이 전체 위험에 똑같이 기여하도록 나눕니다. 특정 종목 쏠림을 막아줍니다.",
    },
    "target_return": {
        "label": "목표수익률 달성",
        "description": "원하는 연 수익률을 정하면 그 수익률을 내면서 위험이 가장 낮은 조합을 찾습니다.",
    },
    "equal_weight": {
        "label": "동일비중",
        "description": "모든 종목에 똑같이 나눕니다. 비교 기준으로 삼기 좋은 가장 단순한 방법입니다.",
    },
}


@dataclass
class OptimizationResult:
    """최적화 결과 한 건."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    method: str
    message: str = ""
    diversification: dict[str, float] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        """비중을 보기 좋은 표로 변환한다."""
        frame = self.weights.rename("비중").to_frame()
        frame["비중(%)"] = (frame["비중"] * 100).round(2)
        return frame.sort_values("비중", ascending=False)


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """포트폴리오 기대수익률."""
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """포트폴리오 변동성(표준편차)."""
    return float(np.sqrt(max(weights @ cov @ weights, 0.0)))


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """종목별 위험기여도. 합은 1이다."""
    total_var = weights @ cov @ weights
    if total_var <= 0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights * (cov @ weights) / total_var


def _check_bounds(n_assets: int, min_weight: float, max_weight: float) -> None:
    """비중 상·하한이 '합이 1'과 양립하는지 확인한다."""
    if not 0.0 <= min_weight <= max_weight <= 1.0:
        raise ValueError("비중 하한과 상한은 0 이상 1 이하이고 하한 <= 상한이어야 합니다.")
    if n_assets * max_weight < 1.0 - 1e-9:
        raise ValueError(
            f"종목 {n_assets}개에 상한 {max_weight:.0%}로는 비중 합 100%를 만들 수 없습니다. "
            f"상한을 {1.0 / n_assets:.0%} 이상으로 올리거나 종목을 더 담아주세요."
        )
    if n_assets * min_weight > 1.0 + 1e-9:
        raise ValueError(
            f"종목 {n_assets}개에 하한 {min_weight:.0%}면 비중 합이 100%를 넘습니다. "
            f"하한을 {1.0 / n_assets:.0%} 이하로 낮춰주세요."
        )


def _starting_points(n_assets: int, min_weight: float, max_weight: float, tries: int) -> list[np.ndarray]:
    """동일비중 + 재현 가능한 난수 시작점들.

    SLSQP 는 지역해에 갇힐 수 있으므로 여러 시작점에서 풀고 가장 좋은 해를 고른다.
    시드를 고정해 같은 입력이면 항상 같은 결과가 나오게 한다.
    """
    points = [np.full(n_assets, 1.0 / n_assets)]
    rng = np.random.default_rng(20240101)
    for _ in range(max(tries - 1, 0)):
        raw = rng.random(n_assets)
        weights = np.clip(raw / raw.sum(), min_weight, max_weight)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        points.append(weights)
    return points


def _solve(
    objective,
    n_assets: int,
    min_weight: float,
    max_weight: float,
    extra_constraints: list[dict] | None = None,
    tries: int = 8,
) -> tuple[np.ndarray | None, str]:
    """제약 하에서 ``objective`` 를 최소화한다. 실패하면 ``(None, 사유)``."""
    bounds = [(min_weight, max_weight)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    if extra_constraints:
        constraints.extend(extra_constraints)

    best_weights, best_value, last_error = None, np.inf, ""
    for start in _starting_points(n_assets, min_weight, max_weight, tries):
        outcome = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if outcome.success and np.isfinite(outcome.fun) and outcome.fun < best_value:
            best_weights, best_value = outcome.x, float(outcome.fun)
        elif not outcome.success:
            last_error = str(outcome.message)

    if best_weights is None:
        return None, last_error or "최적해를 찾지 못했습니다."
    # 부동소수 오차로 상·하한을 미세하게 벗어나는 경우를 정리한다.
    cleaned = np.clip(best_weights, min_weight, max_weight)
    return cleaned / cleaned.sum(), ""


def _risk_parity_weights(
    cov: np.ndarray, min_weight: float, max_weight: float
) -> tuple[np.ndarray, str]:
    """위험균등 비중을 구한다.

    위험기여도 제곱오차를 그대로 최소화하면 목적함수가 비볼록이라, 음의 상관을
    가진 종목의 비중을 0으로 밀어버리는 구석해에 갇히기 쉽다. 대신 아래의
    볼록한 로그배리어 형태를 푼다.

        minimize  ½·wᵀΣw − (1/n)·Σ log(wᵢ)     (w > 0)

    1차 조건이 ``wᵢ(Σw)ᵢ = 1/n`` 이므로 해를 정규화하면 위험기여도가 정확히
    균등해진다. 상·하한을 벗어나는 경우에만 제곱오차 형태로 다시 푼다.
    """
    n_assets = len(cov)

    def objective(w: np.ndarray) -> float:
        return 0.5 * float(w @ cov @ w) - float(np.sum(np.log(w))) / n_assets

    def gradient(w: np.ndarray) -> np.ndarray:
        return cov @ w - 1.0 / (n_assets * w)

    outcome = minimize(
        objective,
        np.full(n_assets, 1.0 / np.sqrt(n_assets)),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(1e-10, None)] * n_assets,
        options={"maxiter": 2000, "ftol": 1e-18, "gtol": 1e-12},
    )
    if outcome.success and np.all(outcome.x > 0):
        weights = outcome.x / outcome.x.sum()
        within_bounds = (weights >= min_weight - 1e-8).all() and (
            weights <= max_weight + 1e-8
        ).all()
        if within_bounds:
            clipped = np.clip(weights, min_weight, max_weight)
            return clipped / clipped.sum(), ""

    # 상·하한에 걸렸다. 정확한 균등은 포기하고 제약 안에서 가장 가깝게 맞춘다.
    target_rc = 1.0 / n_assets
    fallback, error = _solve(
        lambda w: float(np.sum((risk_contributions(w, cov) - target_rc) ** 2)),
        n_assets, min_weight, max_weight, tries=12,
    )
    if fallback is None:
        return np.full(n_assets, 1.0 / n_assets), error
    return fallback, "비중 상·하한 때문에 위험기여도를 완전히 균등하게 맞추지는 못했습니다."


def _build_result(
    weights: np.ndarray,
    labels: pd.Index,
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float,
    method: str,
    message: str = "",
) -> OptimizationResult:
    """가중치 벡터를 성과 지표까지 채운 결과 객체로 만든다."""
    expected = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    sharpe = (expected - risk_free) / vol if vol > 0 else float("nan")
    effective_n = 1.0 / float(np.sum(weights**2)) if np.sum(weights**2) > 0 else float("nan")
    return OptimizationResult(
        weights=pd.Series(weights, index=labels, name="비중"),
        expected_return=expected,
        volatility=vol,
        sharpe=sharpe,
        method=method,
        message=message,
        diversification={
            "유효종목수": effective_n,
            "최대비중": float(weights.max()),
        },
    )


def optimize(
    mu: pd.Series,
    cov: pd.DataFrame,
    method: str = "max_sharpe",
    risk_free: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    target_return: float | None = None,
) -> OptimizationResult:
    """전략에 맞는 최적 비중을 계산한다.

    Args:
        mu: 종목별 연환산 기대수익률.
        cov: 연환산 공분산 행렬. 인덱스/컬럼 순서는 ``mu`` 와 같아야 한다.
        method: :data:`STRATEGIES` 의 키.
        risk_free: 연율 무위험수익률.
        min_weight / max_weight: 종목별 비중 상·하한.
        target_return: ``method="target_return"`` 일 때 목표 연수익률.
    """
    if method not in STRATEGIES:
        raise ValueError(f"알 수 없는 전략입니다: {method}")
    labels = mu.index
    if list(cov.index) != list(labels) or list(cov.columns) != list(labels):
        raise ValueError("기대수익률과 공분산 행렬의 종목 순서가 일치해야 합니다.")

    n_assets = len(labels)
    if n_assets == 0:
        raise ValueError("종목이 하나도 없습니다.")
    if n_assets == 1:
        return _build_result(
            np.array([1.0]), labels, mu.to_numpy(), cov.to_numpy(), risk_free, method,
            "종목이 하나뿐이라 분산 효과가 없습니다.",
        )
    _check_bounds(n_assets, min_weight, max_weight)

    mu_vec, cov_mat = mu.to_numpy(dtype=float), cov.to_numpy(dtype=float)
    equal = np.full(n_assets, 1.0 / n_assets)

    if method == "equal_weight":
        return _build_result(equal, labels, mu_vec, cov_mat, risk_free, method)

    if method == "min_variance":
        objective = lambda w: w @ cov_mat @ w  # noqa: E731
        extra = None
    elif method == "max_sharpe":
        def objective(w: np.ndarray) -> float:
            vol = portfolio_volatility(w, cov_mat)
            if vol <= 1e-12:
                return 1e6
            return -(portfolio_return(w, mu_vec) - risk_free) / vol
        extra = None
    elif method == "risk_parity":
        weights, note = _risk_parity_weights(cov_mat, min_weight, max_weight)
        return _build_result(weights, labels, mu_vec, cov_mat, risk_free, method, note)
    elif method == "target_return":
        if target_return is None:
            raise ValueError("목표수익률 전략에는 target_return 값이 필요합니다.")
        feasible_low, feasible_high = _feasible_return_range(mu_vec, min_weight, max_weight)
        if not feasible_low - 1e-9 <= target_return <= feasible_high + 1e-9:
            raise ValueError(
                f"목표수익률 {target_return:.1%}는 현재 종목 구성으로 달성할 수 없습니다. "
                f"{feasible_low:.1%} ~ {feasible_high:.1%} 사이에서 정해주세요."
            )
        objective = lambda w: w @ cov_mat @ w  # noqa: E731
        extra = [{"type": "eq", "fun": lambda w: float(w @ mu_vec - target_return)}]
    else:  # pragma: no cover - STRATEGIES 검증에서 이미 걸러진다
        raise ValueError(f"구현되지 않은 전략입니다: {method}")

    weights, error = _solve(objective, n_assets, min_weight, max_weight, extra)
    if weights is None:
        return _build_result(
            equal, labels, mu_vec, cov_mat, risk_free, method,
            f"최적화가 수렴하지 않아 동일비중으로 대체했습니다. ({error})",
        )
    return _build_result(weights, labels, mu_vec, cov_mat, risk_free, method)


def _feasible_return_range(
    mu: np.ndarray, min_weight: float, max_weight: float
) -> tuple[float, float]:
    """비중 제약 하에서 도달 가능한 기대수익률의 최소·최대.

    상한이 있으면 수익률이 높은 종목부터(또는 낮은 종목부터) 상한까지 채우는
    탐욕적 배분이 각각 최대·최소가 된다.
    """
    def extreme(order: np.ndarray) -> float:
        weights = np.full(len(mu), min_weight)
        remaining = 1.0 - weights.sum()
        for idx in order:
            add = min(max_weight - weights[idx], remaining)
            weights[idx] += add
            remaining -= add
            if remaining <= 1e-12:
                break
        return float(weights @ mu)

    return extreme(np.argsort(mu)), extreme(np.argsort(mu)[::-1])


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 40,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """효율적 투자선 위의 점들을 계산한다.

    Returns:
        ``수익률`` / ``변동성`` 컬럼을 가진 DataFrame. 변동성 기준 오름차순.
    """
    n_assets = len(mu)
    if n_assets < 2:
        return pd.DataFrame(columns=["수익률", "변동성"])
    _check_bounds(n_assets, min_weight, max_weight)

    mu_vec, cov_mat = mu.to_numpy(dtype=float), cov.to_numpy(dtype=float)
    low, high = _feasible_return_range(mu_vec, min_weight, max_weight)
    # 최소분산 지점보다 아래는 비효율 구간이므로 거기서부터 그린다.
    min_var = optimize(mu, cov, "min_variance", min_weight=min_weight, max_weight=max_weight)
    low = max(low, min_var.expected_return)
    if high - low < 1e-9:
        return pd.DataFrame(
            [{"수익률": min_var.expected_return, "변동성": min_var.volatility}]
        )

    rows = []
    for target in np.linspace(low, high, n_points):
        weights, _ = _solve(
            lambda w: w @ cov_mat @ w,
            n_assets,
            min_weight,
            max_weight,
            [{"type": "eq", "fun": lambda w, t=target: float(w @ mu_vec - t)}],
            tries=3,
        )
        if weights is None:
            continue
        rows.append(
            {
                "수익률": portfolio_return(weights, mu_vec),
                "변동성": portfolio_volatility(weights, cov_mat),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["수익률", "변동성"])
    return pd.DataFrame(rows).sort_values("변동성").reset_index(drop=True)


def discrete_allocation(
    weights: pd.Series, latest_prices: pd.Series, budget: float
) -> pd.DataFrame:
    """비중을 실제 매수 수량(정수 주)으로 환산한다.

    목표 금액을 넘지 않도록 내림으로 배분한 뒤, 남은 현금으로 목표 대비 가장
    많이 미달한 종목부터 한 주씩 더 담는다.
    """
    common = [s for s in weights.index if s in latest_prices.index]
    if not common:
        raise ValueError("비중과 가격 정보가 겹치는 종목이 없습니다.")
    weights = weights.loc[common]
    prices = latest_prices.loc[common].astype(float)
    if (prices <= 0).any():
        raise ValueError("가격이 0 이하인 종목이 있어 수량을 계산할 수 없습니다.")

    target_value = weights * budget
    shares = np.floor(target_value / prices).astype(int)
    cash = float(budget - (shares * prices).sum())

    # 남은 현금으로 목표 대비 가장 부족한 종목부터 채운다.
    while True:
        shortfall = (target_value - shares * prices) / budget
        affordable = [s for s in common if prices[s] <= cash]
        if not affordable:
            break
        pick = max(affordable, key=lambda s: shortfall[s])
        if shortfall[pick] <= 0:
            break
        shares[pick] += 1
        cash -= float(prices[pick])

    actual_value = shares * prices
    total_invested = float(actual_value.sum())
    frame = pd.DataFrame(
        {
            "현재가": prices,
            "목표비중(%)": (weights * 100).round(2),
            "매수수량(주)": shares,
            "매수금액": actual_value.round(0),
            "실제비중(%)": (actual_value / total_invested * 100).round(2)
            if total_invested > 0
            else 0.0,
        }
    )
    frame.attrs["잔여현금"] = float(budget - total_invested)
    return frame
