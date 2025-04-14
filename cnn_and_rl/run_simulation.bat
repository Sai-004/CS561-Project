@echo off
setlocal enabledelayedexpansion

set count=50000
set current=1

:loop
echo Running iteration !current! of %count%
python world3.py

REM Optional: Add a small delay between runs
timeout /t 1 /nobreak > nul

set /a current+=1
if !current! LEQ %count% goto loop

echo Completed all %count% iterations
pause
