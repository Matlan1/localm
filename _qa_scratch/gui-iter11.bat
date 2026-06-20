@echo off
set LOCALM_HOME=D:\projects\localm-qa-home-iter11
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d D:\projects\localm\.claude\worktrees\elated-jennings-330af2
D:\projects\localm\.venv\Scripts\python.exe -m localm gui m8 -p 8645 --no-browser
