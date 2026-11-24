@echo off
rem WL111 W4 LR3 chain on Main. Waits until the fixture chain has finished (sequential GPU use), then runs every LR3 step.
set O=C:\Projects\bench111\lr3\out
set ST=%O%\lr3_chain.status
echo %DATE% %TIME% LR3 CHAIN START commit=%1 > %ST%
:wait
findstr /c:"run111 finished" C:\Projects\bench111\evidence\chain.status >nul 2>&1 && goto go
ping -n 31 127.0.0.1 >nul
goto wait
:go
echo %DATE% %TIME% fixture chain finished, LR3 starts >> %ST%
call C:\Projects\bench111\code\lr3.cmd %O%\pt.out lr3_111.py %O% pt
echo %DATE% %TIME% pt done >> %ST%
for %%R in (eval_wl102_repro eval_L40_adopted eval_L40_fastmatched) do (
  call C:\Projects\bench111\code\lr3e.cmd %O%\%%R.out tg_run.py %%R
  echo %DATE% %TIME% %%R done >> %ST%
)
for %%R in (bench_L40_640 bench_L40_1080) do (
  call C:\Projects\bench111\code\lr3.cmd %O%\%%R.out tg_run.py %%R
  echo %DATE% %TIME% %%R done >> %ST%
)
call C:\Projects\bench111\code\lr3e.cmd %O%\score.out lr3_111.py %O% score
echo %DATE% %TIME% LR3 CHAIN DONE >> %ST%
