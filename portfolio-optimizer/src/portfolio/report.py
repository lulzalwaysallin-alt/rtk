"""분석 결과를 엑셀 파일로 내보낸다.

본부장님이 앱을 닫은 뒤에도 결과를 보관하거나 메일로 돌릴 수 있도록,
화면에 보이는 내용을 시트별로 정리해 하나의 xlsx 로 만든다.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

#: 시트 이름은 31자 제한이 있고 일부 특수문자를 쓸 수 없다.
SHEET_ORDER = ["요약", "추천비중", "매수계획", "성과비교", "상관관계", "가격데이터"]


def _autofit(worksheet, frame: pd.DataFrame, index_label: str | None) -> None:
    """열 너비를 내용 길이에 맞춘다. 한글은 폭을 넓게 잡는다."""
    def width(value) -> int:
        text = str(value)
        korean = sum(1 for ch in text if "가" <= ch <= "힣")
        return len(text) + korean

    first = max([width(index_label or "")] + [width(v) for v in frame.index[:200]])
    worksheet.set_column(0, 0, min(max(first + 3, 12), 40))
    for position, column in enumerate(frame.columns, start=1):
        longest = max([width(column)] + [width(v) for v in frame[column].head(200)])
        worksheet.set_column(position, position, min(max(longest + 3, 10), 30))


def build_excel(
    summary: dict[str, object],
    weights: pd.DataFrame,
    comparison: pd.DataFrame,
    correlation: pd.DataFrame,
    prices: pd.DataFrame,
    purchase_plan: pd.DataFrame | None = None,
) -> bytes:
    """엑셀 파일을 바이트로 만들어 돌려준다. 다운로드 버튼에 그대로 넘기면 된다."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        book = writer.book
        header = book.add_format(
            {"bold": True, "bg_color": "#1e293b", "font_color": "#ffffff",
             "border": 1, "align": "center", "valign": "vcenter"}
        )
        note = book.add_format({"italic": True, "font_color": "#64748b", "font_size": 9})

        summary_frame = pd.DataFrame(
            {"항목": list(summary.keys()), "값": [str(v) for v in summary.values()]}
        ).set_index("항목")
        sheets: dict[str, pd.DataFrame] = {
            "요약": summary_frame,
            "추천비중": weights,
            "성과비교": comparison,
            "상관관계": correlation.round(3),
            "가격데이터": prices.round(2),
        }
        if purchase_plan is not None and not purchase_plan.empty:
            sheets["매수계획"] = purchase_plan

        for name in SHEET_ORDER:
            frame = sheets.get(name)
            if frame is None or frame.empty:
                continue
            index_label = frame.index.name or ("날짜" if name == "가격데이터" else "구분")
            frame.to_excel(writer, sheet_name=name, index=True, index_label=index_label)
            worksheet = writer.sheets[name]
            worksheet.write(0, 0, index_label, header)
            for position, column in enumerate(frame.columns, start=1):
                worksheet.write(0, position, str(column), header)
            worksheet.freeze_panes(1, 1)
            _autofit(worksheet, frame, index_label)

        if "요약" in writer.sheets:
            writer.sheets["요약"].write(
                len(summary_frame) + 3, 0,
                "본 자료는 과거 데이터에 근거한 참고용 분석이며 투자 권유가 아닙니다. "
                "투자 판단과 그 결과에 대한 책임은 투자자 본인에게 있습니다.",
                note,
            )
    return buffer.getvalue()


def default_filename(prefix: str = "포트폴리오분석") -> str:
    """오늘 날짜가 붙은 파일 이름."""
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M}.xlsx"
