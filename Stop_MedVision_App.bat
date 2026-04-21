@echo off
echo Stopping MedVision AI Server...
FOR /F "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo Terminating Process ID %%a
    taskkill /F /PID %%a
)
echo Server Stopped.
pause
