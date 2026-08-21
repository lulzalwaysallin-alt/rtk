"""주가 데이터 수집 계층.

한국 주식은 FinanceDataReader, 미국 주식은 FinanceDataReader → yfinance 순으로
시도한다. 한쪽이 실패해도 다른 쪽으로 넘어가며, 둘 다 실패하면 사람이 읽을 수
있는 한국어 메시지로 예외를 던진다.

내려받은 데이터는 디스크에 캐시한다. 장중에 여러 번 눌러도 매번 외부 서버를
때리지 않도록 하기 위함이다.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MARKET_KR = "KR"
MARKET_US = "US"

CURRENCY_BY_MARKET = {MARKET_KR: "KRW", MARKET_US: "USD"}

#: 캐시 보관 위치와 유효시간(초). 환경변수로 덮어쓸 수 있다.
CACHE_DIR = Path(
    os.environ.get("PORTFOLIO_CACHE_DIR", Path.home() / ".cache" / "portfolio-optimizer")
)
CACHE_TTL_SECONDS = int(os.environ.get("PORTFOLIO_CACHE_TTL", 60 * 60 * 6))

#: 벤치마크 지수. 값은 (FinanceDataReader 코드, yfinance 코드, 통화)
BENCHMARKS: dict[str, tuple[str, str, str]] = {
    "코스피": ("KS11", "^KS11", "KRW"),
    "코스닥": ("KQ11", "^KQ11", "KRW"),
    "S&P 500": ("US500", "^GSPC", "USD"),
    "나스닥 100": ("NASDAQ100", "^NDX", "USD"),
}

#: 처음 쓰는 사람이 바로 눌러볼 수 있는 예시 묶음
PRESETS: dict[str, list[str]] = {
    "국내 대형주 5": ["005930", "000660", "373220", "005380", "051910"],
    "국내 배당·경기방어": ["033780", "017670", "015760", "316140", "086790"],
    "미국 빅테크 5": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "미국 ETF 분산": ["SPY", "QQQ", "TLT", "GLD", "VNQ"],
    "한미 혼합": ["005930", "000660", "AAPL", "MSFT", "SPY"],
}


@dataclass(frozen=True)
class Asset:
    """조회 대상 종목 하나."""

    symbol: str
    name: str
    market: str

    @property
    def currency(self) -> str:
        return CURRENCY_BY_MARKET[self.market]

    @property
    def label(self) -> str:
        """차트·표에 쓸 표시 이름."""
        return f"{self.name}({self.symbol})" if self.name != self.symbol else self.symbol


class DataFetchError(RuntimeError):
    """모든 데이터 소스가 실패했을 때 던진다."""


# --------------------------------------------------------------------------
# 캐시
# --------------------------------------------------------------------------
def _cache_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return CACHE_DIR / f"{digest}.pkl"


def _cache_get(key: str):
    """캐시가 살아 있으면 값을, 아니면 ``None`` 을 돌려준다."""
    path = _cache_path(key)
    try:
        if not path.exists() or time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
            return None
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # 캐시 문제로 앱이 죽어서는 안 된다
        logger.warning("캐시 읽기 실패(무시하고 새로 받습니다): %s", exc)
        return None


def _cache_put(key: str, value) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with _cache_path(key).open("wb") as handle:
            pickle.dump(value, handle)
    except Exception as exc:
        logger.warning("캐시 쓰기 실패(무시합니다): %s", exc)


def clear_cache() -> int:
    """캐시 파일을 모두 지우고 지운 개수를 돌려준다."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for path in CACHE_DIR.glob("*.pkl"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


# --------------------------------------------------------------------------
# 종목 해석
# --------------------------------------------------------------------------
def detect_market(symbol: str) -> str:
    """심볼 모양으로 시장을 판별한다. 6자리 숫자면 한국으로 본다."""
    cleaned = symbol.strip().upper()
    if cleaned.endswith((".KS", ".KQ")):
        return MARKET_KR
    return MARKET_KR if cleaned.isdigit() and len(cleaned) == 6 else MARKET_US


def load_krx_listing() -> pd.DataFrame:
    """KRX 상장 종목 목록(코드·이름·시장)을 받아온다.

    실패하면 빈 DataFrame 을 돌려준다. 이름 검색만 못 할 뿐 코드로는 조회할 수
    있으므로 앱 전체를 멈추지는 않는다.
    """
    cached = _cache_get("krx_listing_v1")
    if cached is not None:
        return cached
    try:
        import FinanceDataReader as fdr

        listing = fdr.StockListing("KRX")
        columns = {c.lower(): c for c in listing.columns}
        code_col = columns.get("code") or columns.get("symbol")
        name_col = columns.get("name")
        market_col = columns.get("market")
        if not code_col or not name_col:
            raise ValueError("상장목록 형식이 예상과 다릅니다.")
        frame = pd.DataFrame(
            {
                "code": listing[code_col].astype(str).str.zfill(6),
                "name": listing[name_col].astype(str),
                "market": listing[market_col].astype(str) if market_col else "",
            }
        ).dropna(subset=["code", "name"])
        _cache_put("krx_listing_v1", frame)
        return frame
    except Exception as exc:
        logger.warning("KRX 상장목록을 받지 못했습니다: %s", exc)
        return pd.DataFrame(columns=["code", "name", "market"])


def search_assets(query: str, limit: int = 20) -> list[Asset]:
    """한글 종목명 또는 코드로 국내 종목을 검색한다.

    정확히 일치 → 코드 일치 → 앞부분 일치 → 부분 일치 순으로 정렬해 돌려준다.
    """
    query = query.strip()
    if not query:
        return []
    listing = load_krx_listing()
    if listing.empty:
        return []

    lowered = listing["name"].str.lower()
    needle = query.lower()
    exact = listing[lowered == needle]
    by_code = listing[listing["code"].str.startswith(query)] if query.isdigit() else listing.iloc[0:0]
    starts = listing[lowered.str.startswith(needle)]
    contains = listing[lowered.str.contains(needle, regex=False)]

    ordered = (
        pd.concat([exact, by_code, starts, contains]).drop_duplicates(subset="code").head(limit)
    )
    return [Asset(symbol=row.code, name=row.name, market=MARKET_KR) for row in ordered.itertuples()]


def resolve_asset(user_input: str) -> Asset:
    """사용자가 입력한 문자열을 :class:`Asset` 으로 바꾼다.

    ``005930`` / ``삼성전자`` / ``AAPL`` 을 모두 받는다.
    """
    text = user_input.strip()
    if not text:
        raise ValueError("종목을 입력해주세요.")

    if detect_market(text) == MARKET_KR:
        code = text.upper().replace(".KS", "").replace(".KQ", "")
        if code.isdigit():
            listing = load_krx_listing()
            match = listing[listing["code"] == code.zfill(6)]
            name = str(match.iloc[0]["name"]) if not match.empty else code.zfill(6)
            return Asset(symbol=code.zfill(6), name=name, market=MARKET_KR)

    # 한글이 섞여 있으면 이름 검색으로 처리한다.
    if any("가" <= ch <= "힣" for ch in text):
        found = search_assets(text, limit=1)
        if found:
            return found[0]
        raise ValueError(
            f"'{text}' 종목을 찾지 못했습니다. 종목명을 정확히 쓰거나 6자리 코드를 입력해주세요."
        )

    return Asset(symbol=text.upper(), name=text.upper(), market=MARKET_US)


# --------------------------------------------------------------------------
# 시세 조회
# --------------------------------------------------------------------------
def _as_date_string(value: str | date | datetime) -> str:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _extract_close(frame: pd.DataFrame) -> pd.Series:
    """조회 결과에서 종가 컬럼을 뽑는다. 컬럼명이 소스마다 다르다."""
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.droplevel(1, axis=1)
    for candidate in ("Adj Close", "Close", "close", "종가"):
        if candidate in frame.columns:
            series = frame[candidate]
            if isinstance(series, pd.DataFrame):  # 중복 컬럼 방어
                series = series.iloc[:, 0]
            return pd.to_numeric(series, errors="coerce").dropna()
    raise ValueError(f"종가 컬럼을 찾을 수 없습니다. 받은 컬럼: {list(frame.columns)}")


def _normalize(series: pd.Series, name: str) -> pd.Series:
    """인덱스를 tz 없는 날짜로 맞추고 중복일을 제거한다."""
    series = series.copy()
    series.name = name
    index = pd.to_datetime(series.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    series.index = index
    return series[~series.index.duplicated(keep="last")].sort_index()


def _fetch_via_fdr(symbol: str, start: str, end: str) -> pd.Series:
    import FinanceDataReader as fdr

    frame = fdr.DataReader(symbol, start, end)
    if frame is None or frame.empty:
        raise ValueError("빈 응답")
    return _extract_close(frame)


def _fetch_via_yfinance(symbol: str, start: str, end: str, market: str) -> pd.Series:
    import yfinance as yf

    # 야후는 한국 종목에 거래소 접미사를 요구한다. 코스피(.KS)를 먼저 시도한다.
    candidates = [f"{symbol}.KS", f"{symbol}.KQ"] if market == MARKET_KR else [symbol]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            frame = yf.download(
                candidate, start=start, end=end, progress=False, auto_adjust=True, threads=False
            )
            if frame is not None and not frame.empty:
                return _extract_close(frame)
            last_error = ValueError("빈 응답")
        except Exception as exc:
            last_error = exc
    raise ValueError(f"yfinance 조회 실패: {last_error}")


def fetch_close_series(symbol: str, start, end, market: str | None = None) -> pd.Series:
    """종목 하나의 종가 시계열을 받아온다. 소스 실패 시 자동으로 다음 소스를 쓴다."""
    start_s, end_s = _as_date_string(start), _as_date_string(end)
    market = market or detect_market(symbol)
    cache_key = f"close_v2|{symbol}|{market}|{start_s}|{end_s}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    errors: list[str] = []
    for source_name, fetch in (
        ("FinanceDataReader", lambda: _fetch_via_fdr(symbol, start_s, end_s)),
        ("yfinance", lambda: _fetch_via_yfinance(symbol, start_s, end_s, market)),
    ):
        try:
            series = fetch()
            if series.empty:
                raise ValueError("데이터가 비어 있습니다.")
            series = _normalize(series, symbol)
            _cache_put(cache_key, series)
            return series
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            logger.info("%s 로 %s 조회 실패: %s", source_name, symbol, exc)

    raise DataFetchError(
        f"'{symbol}' 시세를 받지 못했습니다. 종목코드와 인터넷 연결을 확인해주세요.\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


def fetch_price_table(assets: list[Asset], start, end) -> tuple[pd.DataFrame, list[str]]:
    """여러 종목의 종가를 한 표로 모은다.

    Returns:
        (가격표, 실패한 종목 설명 목록). 일부가 실패해도 나머지로 진행한다.
        컬럼명은 :attr:`Asset.label` 이고, 모든 종목에 값이 있는 구간만 남긴다.
    """
    if not assets:
        raise ValueError("종목을 한 개 이상 선택해주세요.")

    columns: dict[str, pd.Series] = {}
    failures: list[str] = []
    for asset in assets:
        try:
            columns[asset.label] = fetch_close_series(asset.symbol, start, end, asset.market)
        except Exception as exc:
            failures.append(f"{asset.label}: {exc}")

    if not columns:
        raise DataFetchError(
            "선택한 종목 중 시세를 받을 수 있는 것이 없습니다.\n" + "\n".join(failures)
        )

    table = pd.DataFrame(columns).sort_index()
    # 한국·미국은 휴장일이 달라 한쪽만 값이 있는 날이 생긴다. 빈 칸을 직전 값으로
    # 메우면 그날 수익률이 인위적인 0이 되어 변동성과 상관계수가 실제보다 낮게
    # 나온다. 그래서 메우지 않고, 모든 종목이 실제로 거래된 날만 남긴다.
    table = table.dropna(how="any")
    if table.empty:
        raise DataFetchError(
            "선택한 종목들의 거래일이 겹치지 않습니다. 기간을 늘리거나 종목을 바꿔주세요."
        )
    return table, failures


def fetch_benchmark(name: str, start, end) -> pd.Series:
    """벤치마크 지수 시계열을 받아온다."""
    if name not in BENCHMARKS:
        raise ValueError(f"지원하지 않는 벤치마크입니다: {name}")
    fdr_code, yf_code, _ = BENCHMARKS[name]
    start_s, end_s = _as_date_string(start), _as_date_string(end)
    cache_key = f"bench_v2|{name}|{start_s}|{end_s}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    errors = []
    for code, fetch in (
        (fdr_code, lambda: _fetch_via_fdr(fdr_code, start_s, end_s)),
        (yf_code, lambda: _fetch_via_yfinance(yf_code, start_s, end_s, MARKET_US)),
    ):
        try:
            series = _normalize(fetch(), name)
            _cache_put(cache_key, series)
            return series
        except Exception as exc:
            errors.append(f"{code}: {exc}")

    raise DataFetchError(f"'{name}' 지수를 받지 못했습니다. " + "; ".join(errors))


def fetch_usdkrw(start, end) -> pd.Series:
    """원/달러 환율 시계열."""
    start_s, end_s = _as_date_string(start), _as_date_string(end)
    cache_key = f"fx_v2|{start_s}|{end_s}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    errors = []
    for label, fetch in (
        ("USD/KRW", lambda: _fetch_via_fdr("USD/KRW", start_s, end_s)),
        ("KRW=X", lambda: _fetch_via_yfinance("KRW=X", start_s, end_s, MARKET_US)),
    ):
        try:
            series = _normalize(fetch(), "USDKRW")
            _cache_put(cache_key, series)
            return series
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    raise DataFetchError("원/달러 환율을 받지 못했습니다. " + "; ".join(errors))


def convert_to_krw(prices: pd.DataFrame, assets: list[Asset], usdkrw: pd.Series) -> pd.DataFrame:
    """달러 표시 종목을 원화로 환산한다.

    한·미 종목을 섞으면 통화가 달라 수익률을 그대로 비교할 수 없다. 환율까지
    반영해야 원화 투자자가 실제로 얻는 수익률이 된다.
    """
    usd_labels = [a.label for a in assets if a.market == MARKET_US and a.label in prices.columns]
    if not usd_labels:
        return prices
    fx = usdkrw.reindex(prices.index).ffill().bfill()
    converted = prices.copy()
    converted[usd_labels] = converted[usd_labels].mul(fx, axis=0)
    return converted.dropna(how="any")
