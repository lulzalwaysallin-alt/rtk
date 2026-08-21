@echo off
chcp 65001 >nul
title 포트폴리오 최적화 분석기
cd /d "%~dp0"

echo.
echo  ============================================
echo    포트폴리오 최적화 분석기
echo  ============================================
echo.

REM --- 파이썬이 설치되어 있는지 확인 ---
set PY=
where py >nul 2>&1 && set PY=py -3
if not defined PY (
    where python >nul 2>&1 && set PY=python
)
if not defined PY (
    echo  [설치 필요] 파이썬이 없습니다.
    echo.
    echo   1. https://www.python.org/downloads/ 에 접속합니다.
    echo   2. 노란색 [Download Python] 버튼을 눌러 설치 파일을 받습니다.
    echo   3. 설치 화면 맨 아래 "Add python.exe to PATH" 를 반드시 체크합니다.
    echo   4. 설치가 끝나면 이 파일을 다시 실행합니다.
    echo.
    pause
    exit /b 1
)

REM --- 첫 실행이면 필요한 프로그램을 내려받는다 ---
if not exist ".venv\Scripts\python.exe" (
    echo  처음 실행이라 준비 작업을 합니다. 3~5분 정도 걸립니다.
    echo  이 창을 닫지 말고 기다려 주세요.
    echo.
    %PY% -m venv .venv
    if errorlevel 1 goto :setup_failed
)

if not exist ".installed" (
    .venv\Scripts\python.exe -m pip install --upgrade pip --quiet
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto :setup_failed
    echo ok> .installed
    echo.
    echo  준비가 끝났습니다.
    echo.
)

echo  분석기를 실행합니다. 잠시 후 인터넷 창이 자동으로 열립니다.
echo.
echo  ------------------------------------------------------------
echo   * 사용을 마치시면 이 검은 창을 닫아 주세요.
echo   * 창을 닫으면 분석기도 함께 종료됩니다.
echo  ------------------------------------------------------------
echo.

.venv\Scripts\python.exe -m streamlit run app.py
goto :eof

:setup_failed
echo.
echo  [오류] 준비 작업에 실패했습니다.
echo  회사 네트워크가 외부 접속을 막고 있을 수 있습니다.
echo  이 창에 표시된 내용을 담당자에게 전달해 주세요.
echo.
pause
exit /b 1
