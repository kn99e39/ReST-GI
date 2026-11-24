@echo off
rem WL111 LR3: TransGI steps only (rerun after staging datasets/living_room_3/{scene_manifest,material_tables}.json), then scoring.
set O=C:\Projects\bench111\lr3\out
set ST=%O%\lr3_tg.status
echo %DATE% %TIME% LR3 TransGI rerun start > %ST%
for %%R in (eval_wl102_repro eval_L40_adopted eval_L40_fastmatched) do (
  call C:\Projects\bench111\code\lr3e.cmd %O%\%%R.out tg_run.py %%R
  echo %DATE% %TIME% %%R done >> %ST%
)
for %%R in (bench_L40_640 bench_L40_1080) do (
  call C:\Projects\bench111\code\lr3.cmd %O%\%%R.out tg_run.py %%R
  echo %DATE% %TIME% %%R done >> %ST%
)
call C:\Projects\bench111\code\lr3e.cmd %O%\score.out lr3_111.py %O% score
echo %DATE% %TIME% LR3 TG DONE >> %ST%
