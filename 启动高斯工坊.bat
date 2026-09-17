@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "AERIALGS_HOST=127.0.0.1"
set "AERIALGS_PORT=8765"
echo.
echo [高斯工坊] 正在启动，请等待服务加载完成后再进入网页。
echo [高斯工坊] 请不要关闭本窗口。
echo.

if exist "%~dp0runtime\python\python.exe" (
  set "LAUNCH_ROOT=%~dp0"
  goto PORTABLE
)
if exist "%~dp0dist\AerialGS-Portable\runtime\python\python.exe" (
  echo [AerialGS] Using portable runtime in dist\AerialGS-Portable
  set "LAUNCH_ROOT=%~dp0dist\AerialGS-Portable\"
  goto PORTABLE
)
if exist "%~dp0.venv\Scripts\python.exe" goto DEV
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"
exit /b %ERRORLEVEL%

:DEV
echo [高斯工坊] 开发模式启动中，请稍候……
set "AERIALGS_ROOT=%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0backend\run.py" --host 127.0.0.1 --port 8765
exit /b %ERRORLEVEL%

:PORTABLE
set "AERIALGS_ROOT=%LAUNCH_ROOT%"
if exist "%LAUNCH_ROOT%app\backend\run.py" (
  set "PYTHONPATH=%LAUNCH_ROOT%app"
  set "RUNPY=%LAUNCH_ROOT%app\backend\run.py"
) else (
  set "PYTHONPATH=%LAUNCH_ROOT%"
  set "RUNPY=%LAUNCH_ROOT%backend\run.py"
)
set "PYTHONHOME=%LAUNCH_ROOT%runtime\python"
set "TORCH_EXTENSIONS_DIR=%LAUNCH_ROOT%runtime\torch_extensions"
set "PATH=%LAUNCH_ROOT%runtime\python;%LAUNCH_ROOT%runtime\python\Scripts;%LAUNCH_ROOT%runtime\colmap\bin;%LAUNCH_ROOT%runtime\glomap\bin;%PATH%"
"%LAUNCH_ROOT%runtime\python\python.exe" "%RUNPY%" --host 127.0.0.1 --port 8765
exit /b %ERRORLEVEL%
