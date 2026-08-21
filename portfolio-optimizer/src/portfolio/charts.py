"""Plotly 차트 생성.

앱 화면에 그대로 넣을 수 있는 Figure 를 돌려준다. 색과 글꼴을 한곳에서
관리해 화면 전체가 하나의 디자인으로 보이도록 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import metrics
from .optimize import risk_contributions

#: 한글이 깨지지 않도록 국내 환경에 흔한 글꼴을 우선순위대로 나열한다.
FONT_FAMILY = "Pretendard, 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif"

#: 계열 색상. 색맹 사용자도 구분할 수 있도록 명도 차이를 둔다.
PALETTE = [
    "#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6",
    "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
]
POSITIVE, NEGATIVE, NEUTRAL = "#dc2626", "#2563eb", "#94a3b8"


def _style(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    """모든 차트에 공통 레이아웃을 입힌다."""
    fig.update_layout(
        title=title,
        font=dict(family=FONT_FAMILY, size=13),
        height=height,
        margin=dict(l=20, r=20, t=50 if title else 30, b=20),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)", zeroline=False)
    return fig


def weights_donut(weights: pd.Series, title: str = "추천 비중") -> go.Figure:
    """비중 도넛 차트. 0에 가까운 종목은 표시에서 제외한다."""
    shown = weights[weights > 0.0005].sort_values(ascending=False)
    fig = go.Figure(
        go.Pie(
            labels=shown.index,
            values=shown.values,
            hole=0.55,
            marker=dict(colors=PALETTE[: len(shown)], line=dict(color="#ffffff", width=2)),
            textinfo="label+percent",
            texttemplate="%{label}<br>%{percent:.1%}",
            hovertemplate="%{label}<br>비중 %{percent:.2%}<extra></extra>",
            sort=False,
        )
    )
    fig.update_layout(showlegend=False)
    return _style(fig, height=400, title=title)


def efficient_frontier_chart(
    frontier: pd.DataFrame,
    highlights: dict[str, tuple[float, float]] | None = None,
    assets: pd.DataFrame | None = None,
) -> go.Figure:
    """효율적 투자선.

    Args:
        frontier: ``변동성`` / ``수익률`` 컬럼을 가진 DataFrame.
        highlights: 이름 → (변동성, 수익률). 최적 포트폴리오 위치를 별표로 찍는다.
        assets: 개별 종목의 ``변동성`` / ``수익률``.
    """
    fig = go.Figure()
    if not frontier.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier["변동성"], y=frontier["수익률"],
                mode="lines", name="효율적 투자선",
                line=dict(color="#2563eb", width=3),
                hovertemplate="변동성 %{x:.1%}<br>기대수익 %{y:.1%}<extra></extra>",
            )
        )
    if assets is not None and not assets.empty:
        fig.add_trace(
            go.Scatter(
                x=assets["변동성"], y=assets["수익률"],
                mode="markers+text", name="개별 종목",
                text=assets.index, textposition="top center",
                textfont=dict(size=10, color="#64748b"),
                marker=dict(size=9, color=NEUTRAL, symbol="circle"),
                hovertemplate="%{text}<br>변동성 %{x:.1%}<br>기대수익 %{y:.1%}<extra></extra>",
            )
        )
    for offset, (name, (vol, ret)) in enumerate(sorted((highlights or {}).items())):
        fig.add_trace(
            go.Scatter(
                x=[vol], y=[ret], mode="markers", name=name,
                marker=dict(size=18, symbol="star", color=PALETTE[offset % len(PALETTE)],
                            line=dict(color="#ffffff", width=1.5)),
                hovertemplate=f"{name}<br>변동성 %{{x:.1%}}<br>기대수익 %{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_xaxes(title="위험 (연환산 변동성)", tickformat=".0%")
    fig.update_yaxes(title="기대수익률 (연환산)", tickformat=".0%")
    fig.update_layout(hovermode="closest")
    return _style(fig, height=460, title="위험 대비 수익 지도")


def equity_curves(curves: pd.DataFrame, title: str = "누적 수익 추이") -> go.Figure:
    """누적수익 곡선 비교. 값은 원금 대비 배수(1.0 = 원금)."""
    fig = go.Figure()
    for offset, column in enumerate(curves.columns):
        is_portfolio = offset == 0
        fig.add_trace(
            go.Scatter(
                x=curves.index, y=curves[column], name=column, mode="lines",
                line=dict(
                    color=PALETTE[offset % len(PALETTE)],
                    width=3 if is_portfolio else 2,
                    dash=None if is_portfolio else "dot",
                ),
                hovertemplate=f"{column} %{{y:.3f}}배<extra></extra>",
            )
        )
    fig.add_hline(y=1.0, line=dict(color=NEUTRAL, width=1, dash="dash"))
    fig.update_yaxes(title="원금 대비 (배)")
    return _style(fig, height=420, title=title)


def drawdown_chart(returns: pd.Series, title: str = "고점 대비 하락 폭") -> go.Figure:
    """언더워터 차트. 투자자가 실제로 견뎌야 했던 손실 구간을 보여준다."""
    series = metrics.drawdown_series(returns)
    fig = go.Figure(
        go.Scatter(
            x=series.index, y=series.values, mode="lines", fill="tozeroy",
            line=dict(color=NEGATIVE, width=1.5),
            fillcolor="rgba(37,99,235,0.18)",
            name="하락률",
            hovertemplate="%{y:.1%}<extra></extra>",
        )
    )
    worst = series.min()
    if np.isfinite(worst):
        fig.add_hline(
            y=worst, line=dict(color=POSITIVE, width=1, dash="dash"),
            annotation_text=f"최대낙폭 {worst:.1%}", annotation_position="bottom right",
        )
    fig.update_yaxes(title="전고점 대비", tickformat=".0%")
    return _style(fig, height=300, title=title)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """상관관계 히트맵. 1에 가까울수록 같이 움직여 분산 효과가 작다."""
    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="상관계수"),
            hovertemplate="%{y} ↔ %{x}<br>상관계수 %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest")
    return _style(fig, height=max(320, 60 * len(corr)), title="종목 간 동조화 정도")


def risk_contribution_bar(weights: pd.Series, cov: pd.DataFrame) -> go.Figure:
    """투자 비중과 실제 위험기여도를 나란히 비교한다.

    비중은 작은데 위험기여도가 큰 종목이 숨은 위험 요인이다.
    """
    aligned = weights.reindex(cov.index).fillna(0.0)
    contributions = pd.Series(
        risk_contributions(aligned.to_numpy(dtype=float), cov.to_numpy(dtype=float)),
        index=cov.index,
    )
    order = contributions.sort_values(ascending=True).index
    fig = go.Figure()
    fig.add_trace(
        go.Bar(y=list(order), x=aligned[order].values, name="투자 비중",
               orientation="h", marker_color=NEUTRAL,
               hovertemplate="투자 비중 %{x:.1%}<extra></extra>")
    )
    fig.add_trace(
        go.Bar(y=list(order), x=contributions[order].values, name="위험 기여도",
               orientation="h", marker_color="#2563eb",
               hovertemplate="위험 기여도 %{x:.1%}<extra></extra>")
    )
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_xaxes(title="비율", tickformat=".0%")
    return _style(fig, height=max(300, 55 * len(order)), title="비중 대비 실제 위험 부담")


def weights_over_time(history: pd.DataFrame) -> go.Figure:
    """워크포워드 백테스트에서 비중이 어떻게 바뀌었는지 본다."""
    fig = go.Figure()
    for offset, column in enumerate(history.columns):
        fig.add_trace(
            go.Scatter(
                x=history.index, y=history[column], name=column,
                mode="lines", stackgroup="one",
                line=dict(width=0.5, color=PALETTE[offset % len(PALETTE)]),
                hovertemplate=f"{column} %{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_yaxes(title="비중", tickformat=".0%", range=[0, 1])
    return _style(fig, height=340, title="시점별 비중 변화")


def price_comparison(prices: pd.DataFrame) -> go.Figure:
    """종목별 가격 흐름을 시작일 100 기준으로 겹쳐 본다."""
    normalized = prices / prices.iloc[0] * 100.0
    fig = go.Figure()
    for offset, column in enumerate(normalized.columns):
        fig.add_trace(
            go.Scatter(
                x=normalized.index, y=normalized[column], name=column, mode="lines",
                line=dict(color=PALETTE[offset % len(PALETTE)], width=2),
                hovertemplate=f"{column} %{{y:.1f}}<extra></extra>",
            )
        )
    fig.update_yaxes(title="시작일 = 100")
    return _style(fig, height=380, title="종목별 주가 흐름 (시작일 100 기준)")
