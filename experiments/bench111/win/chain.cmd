@echo off
rem WL111 evidence chain on Main: focused tests, then every fixture stage (resumable; completed outputs are skipped).
set E=C:\Projects\bench111\evidence
if not exist %E% mkdir %E%
echo %DATE% %TIME% CHAIN START commit=%1 > %E%\chain.status
call C:\Projects\bench111\code\py.cmd %E%\tests.out tests111.py %E%\tests111.json
findstr /c:"ALL_PASS True" %E%\tests.out >nul || (echo %DATE% %TIME% TESTS FAILED >> %E%\chain.status & exit /b 1)
echo %DATE% %TIME% tests pass >> %E%\chain.status
call C:\Projects\bench111\code\py.cmd %E%\run_all.out run111.py %E% all
echo %DATE% %TIME% run111 finished >> %E%\chain.status
