"""포트폴리오 최적화 분석기.

계층 구조는 다음과 같다.

``data``
    외부에서 시세를 받아온다. 유일하게 네트워크를 쓰는 모듈이다.
``metrics``
    수익률·위험 지표를 계산한다.
``optimize``
    마코위츠 평균-분산 최적화로 비중을 구한다.
``backtest``
    과거 구간에 대해 리밸런싱을 시뮬레이션한다.
``charts`` / ``report``
    결과를 화면과 엑셀로 표현한다.
"""

__version__ = "1.0.0"

__all__ = ["data", "metrics", "optimize", "backtest", "charts", "report"]
