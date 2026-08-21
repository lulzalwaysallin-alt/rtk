"""분석 전체 흐름 검증. 네트워크 없이 가격표를 직접 넣어 확인한다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.pipeline import AnalysisConfig, analyze, headline_metrics


def test_기본_분석이_모든_산출물을_채운다(sample_prices, sample_benchmark):
    result = analyze(sample_prices, AnalysisConfig(), {"코스피": sample_benchmark})
    assert result.optimal.weights.sum() == pytest.approx(1.0)
    assert not result.frontier.empty
    assert not result.comparison.empty
    assert len(result.asset_points) == 4
    assert result.correlation.shape == (4, 4)
    names = [b.name for b in result.backtests]
    assert "추천 포트폴리오" in names
    assert "동일비중 (비교용)" in names
    assert any(n.startswith("워크포워드") for n in names)
    assert "코스피" in names


def test_핵심지표_네_개가_모두_유한하다(sample_prices):
    values = headline_metrics(analyze(sample_prices, AnalysisConfig()))
    assert set(values) == {"기대수익률", "예상변동성", "샤프지수", "과거최대낙폭"}
    assert all(np.isfinite(v) for v in values.values())
    assert values["과거최대낙폭"] <= 0


def test_투자금액을_주면_매수계획이_나온다(sample_prices):
    result = analyze(sample_prices, AnalysisConfig(budget=30_000_000))
    assert result.purchase_plan is not None
    assert result.purchase_plan["매수금액"].sum() <= 30_000_000
    assert result.purchase_plan.attrs["잔여현금"] >= 0


def test_투자금액이_없으면_매수계획도_없다(sample_prices):
    assert analyze(sample_prices, AnalysisConfig(budget=None)).purchase_plan is None


def test_워크포워드를_끄면_실행하지_않는다(sample_prices):
    result = analyze(sample_prices, AnalysisConfig(walk_forward=False))
    assert not any(b.name.startswith("워크포워드") for b in result.backtests)


def test_기간이_짧으면_이유를_설명하며_거절(sample_prices):
    with pytest.raises(ValueError, match="기간을 최소 3개월"):
        analyze(sample_prices.iloc[:20], AnalysisConfig())


def test_짧은_기간은_경고를_남긴다(sample_prices):
    result = analyze(sample_prices.iloc[:60], AnalysisConfig(walk_forward=False))
    assert any("짧습니다" in w for w in result.warnings)


def test_종목이_없으면_거절():
    with pytest.raises(ValueError, match="한 개 이상"):
        analyze(pd.DataFrame(index=pd.bdate_range("2024-01-01", periods=300)), AnalysisConfig())


def test_한_종목만_있으면_경고하고_전액_배분(sample_prices):
    result = analyze(sample_prices[["안정주"]], AnalysisConfig(walk_forward=False))
    assert result.optimal.weights.tolist() == [1.0]
    assert any("분산 효과" in w for w in result.warnings)


def test_워크포워드가_불가능하면_건너뛰고_알려준다(sample_prices):
    # 학습기간(252일)보다 데이터가 짧게 만든다
    result = analyze(sample_prices.iloc[:200], AnalysisConfig(lookback_days=252))
    assert any("워크포워드 검증을 건너뛰었습니다" in w for w in result.warnings)
    assert not any(b.name.startswith("워크포워드") for b in result.backtests)


def test_벤치마크를_못_받으면_분석은_계속된다(sample_prices):
    broken = pd.Series(dtype=float)
    result = analyze(sample_prices, AnalysisConfig(walk_forward=False), {"코스피": broken})
    assert any("비교 곡선을 만들지 못했습니다" in w for w in result.warnings)
    assert not result.comparison.empty, "벤치마크 실패가 분석 전체를 막으면 안 된다"


@pytest.mark.parametrize(
    "method", ["max_sharpe", "min_variance", "risk_parity", "equal_weight"]
)
def test_모든_전략이_끝까지_동작한다(sample_prices, method):
    result = analyze(sample_prices, AnalysisConfig(method=method, walk_forward=False))
    assert result.optimal.weights.sum() == pytest.approx(1.0)
    assert result.optimal.method == method


def test_목표수익률_전략도_끝까지_동작한다(sample_prices):
    from portfolio import metrics

    achievable = float(metrics.expected_returns(metrics.to_returns(sample_prices)).mean())
    result = analyze(
        sample_prices,
        AnalysisConfig(method="target_return", target_return=achievable, walk_forward=False),
    )
    assert result.optimal.expected_return == pytest.approx(achievable, abs=1e-4)


def test_비중_상한이_결과까지_전달된다(sample_prices):
    result = analyze(sample_prices, AnalysisConfig(max_weight=0.30, walk_forward=False))
    assert result.optimal.weights.max() <= 0.30 + 1e-6


def test_같은_입력이면_같은_결과가_나온다(sample_prices):
    first = analyze(sample_prices, AnalysisConfig(walk_forward=False))
    second = analyze(sample_prices, AnalysisConfig(walk_forward=False))
    assert first.optimal.weights.tolist() == pytest.approx(second.optimal.weights.tolist())


def test_추천_포트폴리오는_동일비중보다_사후_샤프가_낮지_않다(sample_prices):
    """같은 구간으로 최적화했으므로 그 구간 성과는 동일비중 이상이어야 정상이다."""
    result = analyze(sample_prices, AnalysisConfig(risk_free=0.02, walk_forward=False))
    table = result.comparison
    assert table.loc["추천 포트폴리오", "샤프지수"] >= table.loc["동일비중 (비교용)", "샤프지수"] - 0.05
