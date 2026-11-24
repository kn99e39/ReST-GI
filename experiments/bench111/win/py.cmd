@echo off
rem WL111 wrapper: py.cmd LOGFILE script args...   (runs in C:\Projects\bench111\code with the frozen ECRO runtime env)
set RESTGI_REPO=C:\Projects\ReST-GI
set PYTHONUTF8=1
set FREEZE=1
set OMP_NUM_THREADS=4
cd /d C:\Projects\bench111\code
set LOGF=%1
shift
C:\Projects\ReST-GI\external\8dna26\.venv-cu128\Scripts\python.exe -B %1 %2 %3 %4 %5 %6 > %LOGF% 2>&1
echo EXIT %ERRORLEVEL% >> %LOGF%
