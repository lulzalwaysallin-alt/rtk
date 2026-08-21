#!/usr/bin/env bash
# 포트폴리오 최적화 분석기 실행 (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"

echo
echo " ============================================"
echo "   포트폴리오 최적화 분석기"
echo " ============================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo " [설치 필요] 파이썬이 없습니다."
    echo "   https://www.python.org/downloads/ 에서 설치한 뒤 다시 실행해 주세요."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo " 처음 실행이라 준비 작업을 합니다. 3~5분 정도 걸립니다."
    python3 -m venv .venv
fi

if [ ! -f ".installed" ]; then
    .venv/bin/python -m pip install --upgrade pip --quiet
    .venv/bin/python -m pip install -r requirements.txt
    touch .installed
    echo " 준비가 끝났습니다."
    echo
fi

echo " 분석기를 실행합니다. 잠시 후 브라우저가 자동으로 열립니다."
echo " 사용을 마치시면 이 터미널 창에서 Ctrl+C 를 누르거나 창을 닫아 주세요."
echo

exec .venv/bin/python -m streamlit run app.py
