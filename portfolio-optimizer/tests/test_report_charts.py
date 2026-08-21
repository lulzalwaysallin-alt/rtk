"""엑셀 리포트와 차트 생성 검증."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from portfolio import charts, report
from portfolio.pipeline import AnalysisConfig, analyze


@pytest.fixture
def result(sample_prices):
    return analyze(sample_prices, AnalysisConfig(budget=10_000_000, walk_forward=False))


def test_엑셀은_열리는_파일이고_시트가_모두_있다(result):
    payload = report.build_excel(
        summary={"분석일": "2026-08-21", "전략": "최대 샤프지수"},
        weights=result.optimal.as_frame(),
        comparison=result.comparison,
        correlation=result.correlation,
        prices=result.prices,
        purchase_plan=result.purchase_plan,
    )
    assert payload[:2] == b"PK", "xlsx 는 zip 형식이어야 한다"
    sheets = pd.read_excel(io.BytesIO(payload), sheet_name=None)
    assert set(sheets) == {"요약", "추천비중", "매수계획", "성과비교", "상관관계", "가격데이터"}
    assert len(sheets["가격데이터"]) == len(result.prices)
    assert len(sheets["추천비중"]) == len(result.optimal.weights)


def test_매수계획이_없어도_엑셀이_만들어진다(result):
    payload = report.build_excel(
        summary={"분석일": "2026-08-21"},
        weights=result.optimal.as_frame(),
        comparison=result.comparison,
        correlation=result.correlation,
        prices=result.prices,
        purchase_plan=None,
    )
    sheets = pd.read_excel(io.BytesIO(payload), sheet_name=None)
    assert "매수계획" not in sheets
    assert "요약" in sheets


def test_파일이름에_날짜가_들어간다():
    name = report.default_filename()
    assert name.startswith("포트폴리오분석_") and name.endswith(".xlsx")


def test_비중_도넛은_0인_종목을_빼고_그린다(result):
    weights = result.optimal.weights.copy()
    weights.iloc[0] = 0.0
    weights = weights / weights.sum()
    figure = charts.weights_donut(weights)
    assert len(figure.data) == 1
    assert len(figure.data[0].labels) == (weights > 0.0005).sum()


def test_투자선_차트는_곡선과_별표를_모두_담는다(result):
    figure = charts.efficient_frontier_chart(
        result.frontier,
        {"추천": (result.optimal.volatility, result.optimal.expected_return)},
        result.asset_points,
    )
    names = [trace.name for trace in figure.data]
    assert "효율적 투자선" in names
    assert "개별 종목" in names
    assert "추천" in names


def test_투자선_차트는_데이터가_비어도_예외를_내지_않는다():
    figure = charts.efficient_frontier_chart(pd.DataFrame(columns=["수익률", "변동성"]))
    assert figure is not None


def test_누적수익_차트는_종목_수만큼_선을_그린다(result):
    from portfolio.backtest import align_curves

    curves = align_curves(result.backtests)
    figure = charts.equity_curves(curves)
    assert len(figure.data) == len(curves.columns)


def test_낙폭_차트는_음수_영역을_그린다(result):
    figure = charts.drawdown_chart(result.main_backtest.returns)
    assert figure.data[0].fill == "tozeroy"
    assert min(figure.data[0].y) <= 0


def test_상관관계_히트맵_범위는_고정된다(result):
    figure = charts.correlation_heatmap(result.correlation)
    assert figure.data[0].zmin == -1 and figure.data[0].zmax == 1


def test_위험기여도_차트는_비중과_기여도를_함께_보여준다(result):
    figure = charts.risk_contribution_bar(result.optimal.weights, result.covariance)
    assert [trace.name for trace in figure.data] == ["투자 비중", "위험 기여도"]
    assert sum(figure.data[0].x) == pytest.approx(1.0, abs=1e-6)
    assert sum(figure.data[1].x) == pytest.approx(1.0, abs=1e-6)


def test_가격_비교_차트는_시작값이_100이다(result):
    figure = charts.price_comparison(result.prices)
    for trace in figure.data:
        assert trace.y[0] == pytest.approx(100.0)


def test_시점별_비중_차트(sample_prices):
    from portfolio.backtest import run_walk_forward

    walk = run_walk_forward(sample_prices, lookback_days=252, frequency="quarterly")
    figure = charts.weights_over_time(walk.weights_history)
    assert len(figure.data) == len(sample_prices.columns)
    assert figure.layout.yaxis.range == (0, 1)


def test_모든_차트에_한글_글꼴이_지정된다(result):
    figure = charts.weights_donut(result.optimal.weights)
    assert "Malgun Gothic" in figure.layout.font.family


def test_단일종목_상관행렬도_그릴_수_있다():
    corr = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    assert charts.correlation_heatmap(corr) is not None
