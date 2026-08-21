"""데이터 계층 검증.

네트워크를 타지 않는다. 외부 API 가 실제로 돌려주는 모양의 픽스처를 읽어
파싱·정규화·환산 로직만 확인한다.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import FIXTURES

from portfolio import data


@pytest.fixture
def fdr_response() -> pd.DataFrame:
    """FinanceDataReader 가 국내 종목에 대해 돌려주는 형태."""
    return pd.read_csv(FIXTURES / "fdr_kr_005930.csv", index_col=0, parse_dates=True)


@pytest.fixture
def yfinance_response() -> pd.DataFrame:
    """yfinance 가 돌려주는 형태(컬럼이 Price/Ticker 2단 인덱스)."""
    return pd.read_csv(FIXTURES / "yfinance_aapl.csv", header=[0, 1], index_col=0, parse_dates=True)


def test_시장_판별():
    assert data.detect_market("005930") == data.MARKET_KR
    assert data.detect_market(" 000660 ") == data.MARKET_KR
    assert data.detect_market("005930.KS") == data.MARKET_KR
    assert data.detect_market("035720.KQ") == data.MARKET_KR
    assert data.detect_market("AAPL") == data.MARKET_US
    assert data.detect_market("brk-b") == data.MARKET_US
    assert data.detect_market("12345") == data.MARKET_US, "6자리가 아니면 국내로 보지 않는다"


def test_통화와_표시이름():
    korean = data.Asset("005930", "삼성전자", data.MARKET_KR)
    assert korean.currency == "KRW"
    assert korean.label == "삼성전자(005930)"
    foreign = data.Asset("AAPL", "AAPL", data.MARKET_US)
    assert foreign.currency == "USD"
    assert foreign.label == "AAPL", "이름과 코드가 같으면 중복해 쓰지 않는다"


def test_국내_응답에서_종가를_뽑는다(fdr_response):
    close = data._extract_close(fdr_response)
    assert len(close) == 10
    assert close.iloc[0] == 73400
    assert close.iloc[-1] == 71700


def test_해외_응답의_2단_컬럼에서도_종가를_뽑는다(yfinance_response):
    close = data._extract_close(yfinance_response)
    assert len(close) == 10
    assert close.iloc[0] == pytest.approx(184.51)


def test_종가_컬럼이_없으면_어떤_컬럼이었는지_알려준다():
    frame = pd.DataFrame({"Open": [1.0], "Volume": [10]})
    with pytest.raises(ValueError, match="종가 컬럼을 찾을 수 없습니다"):
        data._extract_close(frame)


def test_정규화는_시간대를_떼고_중복일을_정리한다():
    index = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-01"]
    ).tz_localize("Asia/Seoul")
    series = pd.Series([1.0, 2.0, 3.0, 0.5], index=index)
    result = data._normalize(series, "테스트")
    assert result.index.tz is None
    assert len(result) == 3
    assert result.is_monotonic_increasing is False or list(result.index) == sorted(result.index)
    assert result.loc["2024-01-03"] == 3.0, "중복일은 마지막 값을 남긴다"
    assert result.name == "테스트"


def test_해외_티커는_그대로_해석된다():
    asset = data.resolve_asset("  aapl ")
    assert asset == data.Asset("AAPL", "AAPL", data.MARKET_US)


def test_빈_입력은_거절():
    with pytest.raises(ValueError, match="종목을 입력"):
        data.resolve_asset("   ")


def test_국내_코드는_이름을_붙여_해석한다(monkeypatch):
    listing = pd.DataFrame(
        {"code": ["005930", "000660"], "name": ["삼성전자", "SK하이닉스"], "market": ["KOSPI", "KOSPI"]}
    )
    monkeypatch.setattr(data, "load_krx_listing", lambda: listing)
    assert data.resolve_asset("005930") == data.Asset("005930", "삼성전자", data.MARKET_KR)
    assert data.resolve_asset("005930.KS").symbol == "005930", "거래소 접미사도 받아준다"


def test_상장목록을_못_받아도_코드로는_동작한다(monkeypatch):
    monkeypatch.setattr(data, "load_krx_listing", lambda: pd.DataFrame(columns=["code", "name", "market"]))
    asset = data.resolve_asset("005930")
    assert asset.symbol == "005930"
    assert asset.name == "005930", "이름을 모르면 코드를 그대로 쓴다"


def test_한글_이름_검색(monkeypatch):
    listing = pd.DataFrame(
        {
            "code": ["005930", "006400", "000660"],
            "name": ["삼성전자", "삼성SDI", "SK하이닉스"],
            "market": ["KOSPI"] * 3,
        }
    )
    monkeypatch.setattr(data, "load_krx_listing", lambda: listing)
    found = data.search_assets("삼성")
    assert [a.name for a in found] == ["삼성전자", "삼성SDI"]
    assert data.resolve_asset("삼성전자").symbol == "005930"
    # 정확히 일치하는 항목이 부분 일치보다 앞에 온다
    assert data.search_assets("삼성전자")[0].symbol == "005930"


def test_없는_한글_이름은_안내와_함께_거절(monkeypatch):
    monkeypatch.setattr(data, "load_krx_listing", lambda: pd.DataFrame({"code": ["005930"], "name": ["삼성전자"], "market": ["KOSPI"]}))
    with pytest.raises(ValueError, match="찾지 못했습니다"):
        data.resolve_asset("없는회사이름")


def test_빈_검색어는_빈_결과():
    assert data.search_assets("") == []


def test_원화_환산은_해외_종목에만_적용된다():
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"삼성전자(005930)": [70000.0] * 5, "AAPL": [100.0] * 5}, index=index)
    assets = [
        data.Asset("005930", "삼성전자", data.MARKET_KR),
        data.Asset("AAPL", "AAPL", data.MARKET_US),
    ]
    fx = pd.Series([1300.0] * 5, index=index)
    converted = data.convert_to_krw(prices, assets, fx)
    assert converted["삼성전자(005930)"].tolist() == [70000.0] * 5
    assert converted["AAPL"].tolist() == [130000.0] * 5


def test_국내_종목만_있으면_환산은_그대로_통과():
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"삼성전자(005930)": [70000.0] * 3}, index=index)
    assets = [data.Asset("005930", "삼성전자", data.MARKET_KR)]
    result = data.convert_to_krw(prices, assets, pd.Series(dtype=float))
    assert result.equals(prices)


def test_환율에_구멍이_있어도_앞뒤_값으로_메운다():
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"AAPL": [100.0] * 5}, index=index)
    assets = [data.Asset("AAPL", "AAPL", data.MARKET_US)]
    fx = pd.Series([1300.0, None, 1310.0, None, None], index=index)
    converted = data.convert_to_krw(prices, assets, fx)
    assert len(converted) == 5
    assert not converted.isna().any().any()
    assert converted["AAPL"].iloc[1] == 130000.0, "빈 날은 직전 환율을 쓴다"


def test_캐시_저장과_회수():
    data._cache_put("테스트키", {"값": 1})
    assert data._cache_get("테스트키") == {"값": 1}
    assert data._cache_get("없는키") is None
    assert data.clear_cache() >= 1
    assert data._cache_get("테스트키") is None


def test_캐시가_만료되면_무시한다(monkeypatch):
    data._cache_put("만료테스트", "옛날값")
    monkeypatch.setattr(data, "CACHE_TTL_SECONDS", -1)
    assert data._cache_get("만료테스트") is None


def test_예시_묶음은_모두_해석_가능한_심볼이다():
    for name, symbols in data.PRESETS.items():
        assert len(symbols) >= 3, f"{name} 묶음이 너무 작다"
        for symbol in symbols:
            assert data.detect_market(symbol) in (data.MARKET_KR, data.MARKET_US)


def test_벤치마크_목록의_형식():
    for name, entry in data.BENCHMARKS.items():
        assert len(entry) == 3, f"{name} 항목은 (fdr코드, yf코드, 통화) 3개여야 한다"
        assert entry[2] in ("KRW", "USD")


def test_종목_없이_가격표를_요청하면_거절():
    with pytest.raises(ValueError, match="한 개 이상"):
        data.fetch_price_table([], "2024-01-01", "2024-12-31")


def test_가격표는_모든_종목에_값이_있는_구간만_남긴다(monkeypatch):
    """국내·미국은 휴장일이 달라 한쪽만 값이 있는 날이 생긴다."""
    kr_index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    us_index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    series = {
        "005930": pd.Series([70000.0, 71000.0, 72000.0, 73000.0], index=kr_index),
        "AAPL": pd.Series([180.0, 181.0, 182.0, 183.0], index=us_index),
    }
    monkeypatch.setattr(
        data, "fetch_close_series", lambda symbol, start, end, market=None: series[symbol]
    )
    assets = [
        data.Asset("005930", "삼성전자", data.MARKET_KR),
        data.Asset("AAPL", "AAPL", data.MARKET_US),
    ]
    table, failures = data.fetch_price_table(assets, "2024-01-01", "2024-01-05")
    assert failures == []
    # 1/1 은 AAPL 이 없고, 1/5 는 삼성전자가 없다 → 1/2~1/4 만 남는다
    assert list(table.index.strftime("%m-%d")) == ["01-02", "01-03", "01-04"]
    assert not table.isna().any().any()


def test_일부_종목이_실패해도_나머지로_진행한다(monkeypatch):
    good = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2024-01-01", periods=3))

    def fake(symbol, start, end, market=None):
        if symbol == "BAD":
            raise data.DataFetchError("상장폐지되었거나 없는 종목입니다")
        return good

    monkeypatch.setattr(data, "fetch_close_series", fake)
    assets = [data.Asset("AAPL", "AAPL", "US"), data.Asset("BAD", "BAD", "US")]
    table, failures = data.fetch_price_table(assets, "2024-01-01", "2024-01-03")
    assert list(table.columns) == ["AAPL"]
    assert len(failures) == 1 and "BAD" in failures[0]


def test_모든_종목이_실패하면_거절(monkeypatch):
    def always_fail(symbol, start, end, market=None):
        raise data.DataFetchError("연결 실패")

    monkeypatch.setattr(data, "fetch_close_series", always_fail)
    with pytest.raises(data.DataFetchError, match="시세를 받을 수 있는 것이 없습니다"):
        data.fetch_price_table([data.Asset("AAPL", "AAPL", "US")], "2024-01-01", "2024-01-03")


def test_거래일이_전혀_겹치지_않으면_안내한다(monkeypatch):
    series = {
        "A": pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"])),
        "B": pd.Series([1.0, 2.0], index=pd.to_datetime(["2023-01-01", "2023-01-02"])),
    }
    monkeypatch.setattr(
        data, "fetch_close_series", lambda symbol, start, end, market=None: series[symbol]
    )
    assets = [data.Asset("A", "A", "US"), data.Asset("B", "B", "US")]
    with pytest.raises(data.DataFetchError, match="거래일이 겹치지 않습니다"):
        data.fetch_price_table(assets, "2023-01-01", "2024-01-02")


def test_지원하지_않는_벤치마크는_거절():
    with pytest.raises(ValueError, match="지원하지 않는 벤치마크"):
        data.fetch_benchmark("닛케이", "2024-01-01", "2024-12-31")
