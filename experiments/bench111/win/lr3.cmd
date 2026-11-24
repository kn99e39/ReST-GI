@echo off
rem WL111 LR3 wrapper: lr3.cmd LOGFILE script args...  (TransGI reproduction venv + Mitsuba 3.9.1 / Dr.Jit 1.5.0 first on PYTHONPATH)
set TRANSGI_REPO=C:\Projects\TransGI_Reproduction
set TRANSGI_DATA=C:\Projects\bench111\lr3\tgdata
set WL111_LR3=C:\Projects\bench111\lr3
set PYTHONPATH=C:\Projects\ReST-GI\external\mi39_dyn101;C:\Projects\TransGI_Reproduction
set PYTHONUTF8=1
cd /d C:\Projects\bench111\code
set LOGF=%1
shift
C:\Projects\TransGI_Reproduction\.venv\Scripts\python.exe -B %1 %2 %3 %4 %5 > %LOGF% 2>&1
echo EXIT %ERRORLEVEL% >> %LOGF%
