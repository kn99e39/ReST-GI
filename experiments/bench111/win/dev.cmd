@echo off
set RESTGI_REPO=C:\Projects\ReST-GI
set PYTHONUTF8=1
set FREEZE=1
cd /d C:\Projects\bench111\code
C:\Projects\ReST-GI\external\8dna26\.venv-cu128\Scripts\python.exe -B dev111.py C:\Projects\bench111\dev %1 > C:\Projects\bench111\dev\dev_%1.out 2>&1
echo EXIT %ERRORLEVEL% >> C:\Projects\bench111\dev\dev_%1.out
