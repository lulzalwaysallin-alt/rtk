"""포트폴리오 최적화 분석기 — 화면.

실행:  streamlit run app.py

이 파일은 화면 구성과 사용자 입력만 담당한다. 계산은 모두 ``src/portfolio`` 의
모듈이 하며, 여기서는 그 결과를 배치할 뿐이다.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from portfolio import charts, data, report  # noqa: E402
from portfolio.backtest import REBALANCE_FREQUENCIES, align_curves  # noqa: E402
from portfolio.optimize import STRATEGIES  # noqa: E402
from portfolio.pipeline import AnalysisConfig, analyze, headline_metrics  # noqa: E402

APP_TITLE = "포트폴리오 최적화 분석기"

#: 기간 선택지 → 조회 일수
PERIODS = {
    "최근 1년": 365,
    "최근 2년": 365 * 2,
    "최근 3년": 365 * 3,
    "최근 5년": 365 * 5,
    "최근 10년": 365 * 10,
}

DISCLAIMER = (
    "본 자료는 과거 주가 데이터를 통계적으로 분석한 **참고용 결과**이며 특정 종목의 매매를 권유하지 않습니다. "
    "과거 수익률이 미래 수익률을 보장하지 않으며, 투자 판단과 그 결과에 대한 책임은 투자자 본인에게 있습니다."
)

CUSTOM_CSS = """
<style>
  html, body, [class*="css"] {
    font-family: Pretendard, 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
  }
  [data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
  [data-testid="stMetricLabel"] { font-size: 0.95rem; opacity: 0.85; }
  [data-testid="stMetric"] {
    background: rgba(37, 99, 235, 0.06);
    border: 1px solid rgba(37, 99, 235, 0.18);
    border-radius: 12px;
    padding: 14px 18px;
  }
  div.stButton > button[kind="primary"] {
    height: 3rem; font-size: 1.05rem; font-weight: 700; border-radius: 10px;
  }
  .hint {
    background: rgba(148,163,184,0.12); border-left: 4px solid #2563eb;
    padding: 12px 16px; border-radius: 6px; font-size: 0.9rem; line-height: 1.7;
  }
</style>
"""


# --------------------------------------------------------------------------
# 캐시를 입힌 데이터 조회
# --------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_listing() -> pd.DataFrame:
    return data.load_krx_listing()


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_prices(symbols: tuple[str, ...], start: str, end: str):
    assets = [data.resolve_asset(s) for s in symbols]
    table, failures = data.fetch_price_table(assets, start, end)
    return table, failures, assets


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_benchmark(name: str, start: str, end: str) -> pd.Series:
    return data.fetch_benchmark(name, start, end)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_usdkrw(start: str, end: str) -> pd.Series:
    return data.fetch_usdkrw(start, end)


# --------------------------------------------------------------------------
# 사이드바
# --------------------------------------------------------------------------
def pick_symbols() -> list[str]:
    """종목 선택 위젯. 국내는 이름으로, 해외는 티커로 고른다."""
    st.markdown("#### 1단계 · 종목 고르기")

    preset_names = ["직접 고르기"] + list(data.PRESETS)
    preset = st.selectbox(
        "예시 묶음", preset_names,
        help="처음이시라면 예시 묶음을 골라 바로 '분석 실행'을 눌러보세요.",
    )
    if preset != "직접 고르기":
        chosen = data.PRESETS[preset]
        st.caption("선택된 종목: " + ", ".join(chosen))
        return chosen

    listing = cached_listing()
    domestic: list[str] = []
    if listing.empty:
        st.warning("국내 종목 목록을 받지 못했습니다. 아래에 6자리 종목코드를 직접 입력해주세요.")
        typed = st.text_input("국내 종목코드", placeholder="005930, 000660")
        domestic = [c.strip() for c in typed.split(",") if c.strip()]
    else:
        options = [f"{row.name}({row.code})" for row in listing.itertuples()]
        picked = st.multiselect(
            "국내 종목", options,
            help="회사 이름을 입력하면 자동으로 검색됩니다. 예: 삼성전자",
            placeholder="종목명을 입력하세요",
        )
        domestic = [item[item.rfind("(") + 1 : -1] for item in picked]

    foreign_text = st.text_input(
        "해외 종목 (티커)", placeholder="AAPL, MSFT, SPY",
        help="미국 주식·ETF는 티커를 쉼표로 구분해 입력하세요.",
    )
    foreign = [t.strip().upper() for t in foreign_text.split(",") if t.strip()]

    selected = domestic + foreign
    if selected:
        st.caption(f"선택된 종목 {len(selected)}개: " + ", ".join(selected))
    return selected


def pick_settings() -> tuple[str, AnalysisConfig, list[str], bool]:
    """기간·전략·세부설정을 받는다."""
    st.markdown("#### 2단계 · 기간과 전략")
    period = st.selectbox("분석 기간", list(PERIODS), index=2)

    labels = {key: value["label"] for key, value in STRATEGIES.items()}
    method = st.radio(
        "투자 성향",
        list(labels), index=0,
        format_func=lambda key: labels[key],
    )
    st.markdown(f"<div class='hint'>{STRATEGIES[method]['description']}</div>", unsafe_allow_html=True)

    target_return = None
    if method == "target_return":
        target_return = st.slider(
            "목표 연 수익률", min_value=0.0, max_value=0.50, value=0.12, step=0.01,
            format="%.0f%%",
        )

    config = AnalysisConfig(method=method, target_return=target_return)

    with st.expander("세부 설정 (그대로 두셔도 됩니다)"):
        config.max_weight = st.slider(
            "한 종목 최대 비중", 0.10, 1.00, 0.40, 0.05, format="%.0f%%",
            help="한 종목에 쏠리는 것을 막습니다. 낮출수록 고르게 분산됩니다.",
        )
        config.min_weight = st.slider(
            "한 종목 최소 비중", 0.00, 0.20, 0.00, 0.01, format="%.0f%%",
            help="0보다 크게 두면 고른 종목을 반드시 모두 담습니다.",
        )
        config.risk_free = st.slider(
            "무위험수익률 (연)", 0.00, 0.10, 0.03, 0.005, format="%.1f%%",
            help="예금·국채 금리입니다. 샤프지수 계산의 기준선이 됩니다.",
        )
        config.rebalance = st.selectbox(
            "리밸런싱 주기", list(REBALANCE_FREQUENCIES), index=2,
            format_func=lambda key: REBALANCE_FREQUENCIES[key],
            help="정해진 비중으로 되돌리는 주기입니다.",
        )
        config.cost_bps = st.slider(
            "거래비용 (편도, bp)", 0.0, 100.0, 15.0, 5.0,
            help="수수료와 세금입니다. 15bp = 0.15%.",
        )
        config.walk_forward = st.checkbox(
            "워크포워드 검증 함께 실행", value=True,
            help="매 시점에 그때까지의 데이터만으로 다시 계산합니다. 더 현실적이지만 시간이 조금 걸립니다.",
        )
        budget_manwon = st.number_input(
            "투자 예정 금액 (만원)", min_value=0, value=5000, step=100,
            help="입력하면 종목별로 몇 주를 사야 하는지까지 계산합니다. 0이면 생략합니다.",
        )
        config.budget = float(budget_manwon) * 10_000 if budget_manwon > 0 else None

    st.markdown("#### 3단계 · 비교 대상")
    benchmarks = st.multiselect(
        "함께 볼 지수", list(data.BENCHMARKS), default=["코스피", "S&P 500"],
        help="내 포트폴리오가 시장 대비 잘했는지 비교합니다.",
    )
    convert_krw = st.checkbox(
        "해외 종목을 원화로 환산", value=True,
        help="환율 변동까지 반영해 원화 투자자 기준 수익률을 계산합니다.",
    )
    return period, config, benchmarks, convert_krw


# --------------------------------------------------------------------------
# 본문 렌더링
# --------------------------------------------------------------------------
def render_welcome() -> None:
    """분석 전 안내 화면."""
    st.markdown(
        """
        ### 무엇을 해주는 프로그램인가요?

        여러 종목에 **얼마씩 나눠 담아야 위험 대비 수익이 가장 좋은지**를 계산합니다.
        노벨경제학상을 받은 마코위츠의 포트폴리오 이론을 그대로 적용했습니다.

        **사용법은 세 단계입니다.**

        1. 왼쪽에서 **종목을 고릅니다.** 처음이시라면 맨 위 '예시 묶음'을 하나 고르세요.
        2. **기간과 투자 성향**을 정합니다. 기본값 그대로 두셔도 됩니다.
        3. 아래 **[분석 실행]** 버튼을 누릅니다.

        결과로는 종목별 추천 비중, 위험·수익 지도, 과거에 이 비중으로 투자했다면
        어땠을지에 대한 검증, 그리고 실제 매수 수량까지 나옵니다.
        """
    )
    st.info(
        "**한 가지만 기억해주세요.** 이 계산은 과거 데이터에 기반합니다. "
        "'앞으로 이만큼 번다'가 아니라 '과거와 비슷한 흐름이라면 이 비중이 균형적이다'라는 뜻입니다."
    )


def render_headline(result) -> None:
    """상단 핵심 지표 카드."""
    values = headline_metrics(result)
    columns = st.columns(4)
    columns[0].metric(
        "연 기대수익률", f"{values['기대수익률']:.1%}",
        help="과거 평균 수익률을 1년 기준으로 환산한 값입니다.",
    )
    columns[1].metric(
        "연 예상변동성", f"{values['예상변동성']:.1%}",
        help="가격이 위아래로 흔들리는 폭입니다. 낮을수록 안정적입니다.",
    )
    columns[2].metric(
        "샤프지수", f"{values['샤프지수']:.2f}",
        help="위험 한 단위당 초과수익입니다. 1 이상이면 양호, 2 이상이면 우수합니다.",
    )
    columns[3].metric(
        "과거 최대낙폭", f"{values['과거최대낙폭']:.1%}",
        help="분석 기간 중 고점 대비 가장 크게 떨어졌던 폭입니다. 견뎌야 할 손실의 크기입니다.",
    )


def render_weights_tab(result) -> None:
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(charts.weights_donut(result.optimal.weights), width="stretch")
    with right:
        st.markdown("##### 종목별 추천 비중")
        frame = result.optimal.as_frame()[["비중(%)"]]
        frame["연 기대수익률(%)"] = (result.expected.reindex(frame.index) * 100).round(1)
        frame["연 변동성(%)"] = (
            result.asset_points["변동성"].reindex(frame.index) * 100
        ).round(1)
        st.dataframe(frame, width="stretch")
        effective = result.optimal.diversification.get("유효종목수", float("nan"))
        st.caption(
            f"실질 분산 정도: 종목 {len(result.optimal.weights)}개를 담았지만 "
            f"쏠림을 감안하면 **{effective:.1f}개**에 나눠 담은 효과입니다."
        )

    if result.purchase_plan is not None and not result.purchase_plan.empty:
        st.markdown("##### 실제 매수 계획")
        st.caption("마지막 종가 기준입니다. 실제 체결가는 다를 수 있습니다.")
        plan = result.purchase_plan.copy()
        st.dataframe(
            plan.style.format(
                {"현재가": "{:,.0f}", "매수금액": "{:,.0f}", "목표비중(%)": "{:.2f}",
                 "실제비중(%)": "{:.2f}", "매수수량(주)": "{:,.0f}"}
            ),
            width="stretch",
        )
        st.caption(f"남는 현금: {plan.attrs.get('잔여현금', 0):,.0f}원")

    st.plotly_chart(
        charts.risk_contribution_bar(result.optimal.weights, result.covariance),
        width="stretch",
    )
    st.caption(
        "회색(투자 비중)보다 파란색(위험 기여도)이 훨씬 긴 종목은, 적게 담았는데도 "
        "전체 위험을 크게 좌우하는 종목입니다."
    )


def render_map_tab(result) -> None:
    highlights = {
        "추천 포트폴리오": (result.optimal.volatility, result.optimal.expected_return),
        "동일비중": (result.equal_weight.volatility, result.equal_weight.expected_return),
    }
    st.plotly_chart(
        charts.efficient_frontier_chart(result.frontier, highlights, result.asset_points),
        width="stretch",
    )
    st.caption(
        "파란 곡선이 '같은 위험에서 얻을 수 있는 최고 수익' 선입니다. "
        "별표가 곡선 위에 있으면 더 나은 조합이 없다는 뜻입니다."
    )
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(charts.correlation_heatmap(result.correlation), width="stretch")
        st.caption("빨간색(1에 가까움)이 많으면 같이 오르내려 분산 효과가 작습니다.")
    with right:
        st.plotly_chart(charts.price_comparison(result.prices), width="stretch")


def render_backtest_tab(result) -> None:
    curves = align_curves(result.backtests)
    if not curves.empty:
        st.plotly_chart(charts.equity_curves(curves), width="stretch")
        final = curves.iloc[-1].sort_values(ascending=False)
        best = final.index[0]
        st.caption(
            f"같은 날 같은 금액으로 시작했다면, 기간 말 기준 **{best}**가 "
            f"원금의 {final.iloc[0]:.2f}배로 가장 높았습니다."
        )

    st.markdown("##### 성과 비교표")
    display = result.comparison.copy()
    percent_columns = ["연환산수익률", "연환산변동성", "최대낙폭", "누적수익률", "일간VaR95", "일간CVaR95"]
    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(lambda v: f"{v:.2%}" if pd.notna(v) else "-")
    for column in ["샤프지수", "소르티노지수", "칼마지수", "베타"]:
        if column in display:
            display[column] = display[column].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    if "총거래비용" in display:
        display["총거래비용"] = display["총거래비용"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "-")
    st.dataframe(display, width="stretch")

    st.markdown(
        "<div class='hint'>"
        "<b>추천 포트폴리오</b>는 지금 계산한 비중을 과거에 그대로 적용한 결과라 실제보다 좋게 나옵니다. "
        "<b>워크포워드</b>는 매 시점에 그때까지의 데이터만 쓴 결과이므로, 이쪽이 실제 운용에 훨씬 가깝습니다. "
        "두 값의 차이가 크다면 그만큼 과거에 최적화된 결과라는 신호입니다."
        "</div>",
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        charts.drawdown_chart(result.main_backtest.returns), width="stretch"
    )

    walk_forward = next(
        (b for b in result.backtests if b.name.startswith("워크포워드")), None
    )
    if walk_forward is not None and not walk_forward.weights_history.empty:
        st.plotly_chart(
            charts.weights_over_time(walk_forward.weights_history), width="stretch"
        )
        st.caption("워크포워드 검증에서 시점마다 계산된 비중입니다. 자주 크게 바뀐다면 불안정한 조합입니다.")


def render_download_tab(result, meta: dict) -> None:
    st.markdown("##### 분석 결과 내려받기")
    st.caption("아래 버튼을 누르면 모든 표와 데이터가 담긴 엑셀 파일이 저장됩니다.")
    try:
        payload = report.build_excel(
            summary=meta,
            weights=result.optimal.as_frame(),
            comparison=result.comparison,
            correlation=result.correlation,
            prices=result.prices,
            purchase_plan=result.purchase_plan,
        )
        st.download_button(
            "엑셀 파일로 저장", payload, file_name=report.default_filename(),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
    except Exception as exc:
        st.error(f"엑셀 파일을 만들지 못했습니다: {exc}")

    st.markdown("##### 조회한 원본 주가")
    st.dataframe(result.prices.tail(120).sort_index(ascending=False), width="stretch")


# --------------------------------------------------------------------------
# 실행 흐름
# --------------------------------------------------------------------------
def run_analysis(symbols, period, config, benchmark_names, convert_krw):
    """데이터를 받아 분석까지 수행한다. 실패는 화면용 메시지로 바꿔 올린다."""
    end = date.today()
    start = end - timedelta(days=PERIODS[period])
    start_s, end_s = start.isoformat(), end.isoformat()

    prices, failures, assets = cached_prices(tuple(symbols), start_s, end_s)

    if convert_krw and any(a.market == data.MARKET_US for a in assets):
        try:
            prices = data.convert_to_krw(prices, assets, cached_usdkrw(start_s, end_s))
        except Exception as exc:
            failures.append(f"환율을 받지 못해 원화 환산을 생략했습니다: {exc}")

    benchmarks = {}
    for name in benchmark_names:
        try:
            benchmarks[name] = cached_benchmark(name, start_s, end_s)
        except Exception as exc:
            failures.append(f"'{name}' 지수를 받지 못했습니다: {exc}")

    result = analyze(prices, config, benchmarks)
    result.warnings = failures + result.warnings
    return result, start_s, end_s


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title(f"📊 {APP_TITLE}")
    st.caption("여러 종목에 얼마씩 나눠 담아야 위험 대비 수익이 가장 좋은지 계산합니다.")

    with st.sidebar:
        st.markdown(f"### {APP_TITLE}")
        symbols = pick_symbols()
        st.divider()
        period, config, benchmark_names, convert_krw = pick_settings()
        st.divider()
        submitted = st.button("분석 실행", type="primary", width="stretch")
        if st.button("저장된 데이터 지우기", width="stretch",
                     help="시세가 오래되었다고 느껴지면 눌러주세요."):
            st.cache_data.clear()
            removed = data.clear_cache()
            st.success(f"임시 저장된 시세 {removed}건을 지웠습니다. 다시 분석해주세요.")

    if submitted:
        if len(symbols) < 2:
            st.error("종목을 2개 이상 골라주세요. 한 종목만으로는 분산 효과를 계산할 수 없습니다.")
        else:
            with st.spinner("시세를 받아 계산하는 중입니다. 처음 실행은 20초쯤 걸릴 수 있습니다."):
                try:
                    result, start_s, end_s = run_analysis(
                        symbols, period, config, benchmark_names, convert_krw
                    )
                    st.session_state["result"] = result
                    st.session_state["meta"] = {
                        "분석일": date.today().isoformat(),
                        "분석기간": f"{start_s} ~ {end_s} ({period})",
                        "종목": ", ".join(symbols),
                        "전략": STRATEGIES[config.method]["label"],
                        "한 종목 최대 비중": f"{config.max_weight:.0%}",
                        "무위험수익률": f"{config.risk_free:.1%}",
                        "리밸런싱": config.rebalance,
                        "거래비용": f"{config.cost_bps:.0f}bp",
                    }
                except Exception as exc:
                    st.session_state.pop("result", None)
                    st.error(f"분석하지 못했습니다.\n\n{exc}")

    result = st.session_state.get("result")
    if result is None:
        render_welcome()
    else:
        for message in result.warnings:
            st.warning(message)
        render_headline(result)
        st.divider()
        tabs = st.tabs(["추천 비중", "위험·수익 지도", "과거 검증", "저장하기"])
        with tabs[0]:
            render_weights_tab(result)
        with tabs[1]:
            render_map_tab(result)
        with tabs[2]:
            render_backtest_tab(result)
        with tabs[3]:
            render_download_tab(result, st.session_state.get("meta", {}))

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
