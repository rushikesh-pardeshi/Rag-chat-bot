@echo off
rem run.cmd — start the whole stack from cmd.exe.
rem
rem     run            start PostgreSQL + API + UI (Ctrl-C stops API + UI)
rem     run --stop     stop the API and the UI, leave PostgreSQL running
rem
rem start.sh is the real script; cmd.exe cannot execute a .sh file, so typing
rem "start.sh" at a cmd prompt returns to the prompt having started nothing —
rem which is exactly what it looks like when it happens. This wrapper finds Git's
rem bash.exe and hands the script to it.
rem
rem Not named start.cmd on purpose: "start" is a cmd built-in and would win over
rem the file, so the command would open a new window instead of running this.

setlocal
rem Forward slashes matter: start.sh locates itself with dirname, and dirname of a
rem backslash path returns "." — the script would then work only when cmd's current
rem directory already happened to be the project root.
set "HERE=%~dp0"
set "HERE=%HERE:\=/%"
set "BASH="

rem Git for Windows, per-user install (no admin rights) — the layout on this machine.
if exist "%LOCALAPPDATA%\Programs\Git\bin\bash.exe" set "BASH=%LOCALAPPDATA%\Programs\Git\bin\bash.exe"
rem Machine-wide installs.
if not defined BASH if exist "%ProgramFiles%\Git\bin\bash.exe" set "BASH=%ProgramFiles%\Git\bin\bash.exe"
if not defined BASH if exist "%ProgramFiles(x86)%\Git\bin\bash.exe" set "BASH=%ProgramFiles(x86)%\Git\bin\bash.exe"
rem Anything already on PATH, last: it may be a WSL bash, which cannot see these Windows paths.
if not defined BASH for /f "delims=" %%B in ('where bash.exe 2^>nul') do if not defined BASH set "BASH=%%B"

if not defined BASH (
  echo.
  echo FAILED: no bash.exe found. Install Git for Windows, or run start.sh from Git Bash.
  exit /b 1
)

echo Using %BASH%
"%BASH%" "%HERE%start.sh" %*
exit /b %ERRORLEVEL%
