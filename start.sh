#!/usr/bin/env bash
#
# start.sh — bring up the whole local stack: PostgreSQL, the API, the UI.
#
# Run it from the project root in Git Bash, right after editing `.env`:
#
#     ./start.sh            # start everything, follow the logs, Ctrl-C to stop
#     ./start.sh --stop     # stop the API and the UI, leave PostgreSQL up
#
# **Editing `.env` is not enough on its own — this restart is what applies it.**
# Settings are cached at import time and `--reload` does not watch `.env`, so a
# refreshed AWS session token only reaches Bedrock through a new process. That is
# the reason this script exists as one command instead of three terminals.
#
# Nothing here reads or echoes a secret. `.env` is checked for *presence* of a few
# keys and never for their contents, and no credential is ever passed on a command
# line, where it would land in the process list and in shell history.
#
# Ports 8000 and 3000 are reclaimed before binding. A leftover server from an
# earlier run is exactly what produces
# `[WinError 10013] An attempt was made to access a socket in a way forbidden by
# its access permissions` — that error is a busy port, not a permissions problem,
# and diagnosing it by hand costs more than clearing it here.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

# Overridable so a second machine does not have to edit the script. The defaults are
# the local PostgreSQL laid down by `scripts/setup_postgres.py`.
PG_BIN="${PG_BIN:-C:/Users/<windows-user>/pgsql/bin}"
PG_DATA="${PG_DATA:-C:/Users/<windows-user>/pgdata}"
PG_PORT="${PG_PORT:-5432}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"

VENV_PY="$ROOT/.venv/Scripts/python.exe"
LOG_DIR="$ROOT/logs"
API_LOG="$LOG_DIR/api.log"
WEB_LOG="$LOG_DIR/web.log"
PG_LOG="$LOG_DIR/postgres.log"

say() { printf '\n== %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- ports

# PIDs listening on a TCP port. `netstat -ano` columns: proto local foreign state pid.
pids_on_port() {
  netstat -ano | tr -d '\r' |
    awk -v port=":$1" '$1 == "TCP" && $2 ~ port"$" && $4 == "LISTENING" {print $5}' |
    sort -u
}

image_of_pid() {
  tasklist //FI "PID eq $1" //FO CSV //NH 2>/dev/null |
    tr -d '\r' | head -1 | cut -d, -f1 | tr -d '"'
}

# Free a port, but only from a process that plausibly belongs to this stack. Killing
# an unknown listener silently is how a script eats something a developer needed.
reclaim_port() {
  local port="$1" label="$2" pid image
  for pid in $(pids_on_port "$port"); do
    image="$(image_of_pid "$pid")"
    case "$image" in
      python.exe | pythonw.exe | node.exe | npm.exe | cmd.exe | uvicorn.exe)
        info "port $port held by $image (PID $pid) — stopping it ($label)"
        taskkill //PID "$pid" //T //F >/dev/null 2>&1
        ;;
      '')
        info "port $port held by PID $pid (unknown image) — stopping it ($label)"
        taskkill //PID "$pid" //T //F >/dev/null 2>&1
        ;;
      *)
        die "port $port is held by $image (PID $pid), which is not part of this stack.
       Close it, or run with ${label^^}_PORT set to a free port."
        ;;
    esac
  done
  # Windows keeps the socket briefly after the owner dies.
  local waited=0
  while [ -n "$(pids_on_port "$port")" ] && [ "$waited" -lt 20 ]; do
    sleep 0.5
    waited=$((waited + 1))
  done
  [ -z "$(pids_on_port "$port")" ] || die "port $port did not come free."
}

# ------------------------------------------------------------------------- probing

# `-fs`, not `-fsS`: a failed probe is the normal state while a server is still booting,
# and printing curl's diagnostic for each attempt buries the one line that matters.
http_ok() {
  local timeout="${2:-5}"
  if command -v curl >/dev/null 2>&1; then
    curl -fs -o /dev/null --connect-timeout 2 --max-time "$timeout" "$1"
  else
    "$VENV_PY" - "$1" "$timeout" <<'PY' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])).read()
PY
  fi
}

# Poll until something is listening. `alive_pid` lets a crash be reported straight away
# instead of after the full timeout.
#
# Ports first, HTTP second, because `next dev` compiles a route on its first request: the
# port opens in a second or two while `/login` can take half a minute to answer. Treating a
# slow first response as "not started" is what made an already-working UI look dead.
wait_for_port() {
  local port="$1" tries="$2" alive_pid="${3:-}" n=0
  while [ "$n" -lt "$tries" ]; do
    [ -n "$(pids_on_port "$port")" ] && return 0
    if [ -n "$alive_pid" ] && ! kill -0 "$alive_pid" 2>/dev/null; then
      return 1
    fi
    sleep 0.5
    n=$((n + 1))
  done
  return 1
}

wait_for_http() {
  local url="$1" tries="$2" alive_pid="${3:-}" n=0
  while [ "$n" -lt "$tries" ]; do
    http_ok "$url" && return 0
    if [ -n "$alive_pid" ] && ! kill -0 "$alive_pid" 2>/dev/null; then
      return 1
    fi
    sleep 0.5
    n=$((n + 1))
  done
  return 1
}

# The status code itself, `000` if the request never completed. Used where 404 has to be
# told apart from unreachable — `http_ok` collapses both into failure.
http_status() {
  local timeout="${2:-30}"
  if command -v curl >/dev/null 2>&1; then
    curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time "$timeout" "$1" ||
      printf '000'
  else
    "$VENV_PY" - "$1" "$timeout" 2>/dev/null <<'PY' || printf '000'
import sys, urllib.error, urllib.request
try:
    print(urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])).status, end="")
except urllib.error.HTTPError as failure:
    print(failure.code, end="")
PY
  fi
}

# ------------------------------------------------------------------------ shutdown

stop_stack() {
  reclaim_port "$API_PORT" api
  reclaim_port "$WEB_PORT" web
}

shutdown() {
  trap - INT TERM
  say "stopping"
  [ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null
  stop_stack
  info "API and UI stopped. PostgreSQL is still running — stop it with:"
  info "  \"$PG_BIN/pg_ctl.exe\" -D \"$PG_DATA\" stop"
  exit 0
}

if [ "${1:-}" = "--stop" ]; then
  say "stopping API and UI"
  stop_stack
  info "done. PostgreSQL left running."
  exit 0
fi

# ------------------------------------------------------------------------ preflight

say "preflight"
[ -x "$VENV_PY" ] || die "no virtualenv at .venv — run: uv venv && uv pip install -r requirements.txt"
[ -f "$ROOT/.env" ] || die "no .env — run: cp .env.example .env, then fill in the real values"
[ -x "$PG_BIN/pg_ctl.exe" ] || die "no pg_ctl.exe under $PG_BIN — set PG_BIN, or run scripts/setup_postgres.py"
mkdir -p "$LOG_DIR"

# Presence only. The value never leaves the file, so a missing credential is reported
# without putting it in the terminal, in a log, or in the process list.
env_has_value() {
  grep -Eq "^[[:space:]]*$1[[:space:]]*=[[:space:]]*[^[:space:]#]" "$ROOT/.env"
}
missing=""
for key in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY PINECONE_API_KEY PINECONE_INDEX_NAME; do
  env_has_value "$key" || missing="$missing $key"
done
if [ -n "$missing" ]; then
  info "WARNING: empty or absent in .env:$missing"
  info "the API will boot, but embedding and retrieval will fail until they are set"
else
  info ".env carries values for the AWS and Pinecone keys (values not read)"
fi
if env_has_value AWS_SESSION_TOKEN; then
  info "AWS_SESSION_TOKEN is set — these are temporary credentials, so they expire."
  info "ExpiredTokenException in logs/api.log means refresh .env and run this again."
fi

# ------------------------------------------------------------------------- database

say "1/4 PostgreSQL on port $PG_PORT"
"$PG_BIN/pg_ctl.exe" -D "$PG_DATA" status >/dev/null 2>&1
case $? in
  0) info "already running — left as it is" ;;
  3)
    "$PG_BIN/pg_ctl.exe" -D "$PG_DATA" -o "-p $PG_PORT" start >>"$PG_LOG" 2>&1 ||
      die "pg_ctl start failed — see logs/postgres.log"
    info "started (log: logs/postgres.log)"
    ;;
  *) die "pg_ctl cannot read the data directory $PG_DATA — check PG_DATA" ;;
esac

if [ -x "$PG_BIN/pg_isready.exe" ]; then
  n=0
  until "$PG_BIN/pg_isready.exe" -p "$PG_PORT" -q || [ "$n" -ge 30 ]; do
    sleep 0.5
    n=$((n + 1))
  done
  "$PG_BIN/pg_isready.exe" -p "$PG_PORT" -q ||
    die "PostgreSQL is up but not accepting connections on $PG_PORT"
  info "accepting connections"
fi

# ------------------------------------------------------------------------ migrations

say "2/4 schema"
if "$ROOT/.venv/Scripts/alembic.exe" upgrade head >>"$API_LOG" 2>&1; then
  info "alembic upgrade head — at head"
else
  die "alembic upgrade head failed — tail logs/api.log. If the database itself is
       missing, create it first with: .venv/Scripts/python.exe scripts/setup_postgres.py"
fi

# --------------------------------------------------------------------------- backend

say "3/4 API on http://$API_HOST:$API_PORT"
reclaim_port "$API_PORT" api
# dev_server.py, not bare uvicorn: it points TLS at .certs/corporate-ca-bundle.pem and
# strips the AWS_* shell variables that would otherwise shadow .env.
"$VENV_PY" scripts/dev_server.py --host "$API_HOST" --port "$API_PORT" >>"$API_LOG" 2>&1 &
API_PID=$!
if wait_for_port "$API_PORT" 120 "$API_PID" &&
  wait_for_http "http://$API_HOST:$API_PORT/health" 60; then
  info "healthy (log: logs/api.log)"
else
  printf '\n--- last 25 lines of logs/api.log ---\n'
  tail -25 "$API_LOG"
  die "the API did not answer /health."
fi

# -------------------------------------------------------------------------- frontend

say "4/4 UI on http://localhost:$WEB_PORT"
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  info "installing frontend dependencies (first run only)…"
  (cd "$ROOT/frontend" && npm install) >>"$WEB_LOG" 2>&1 ||
    die "npm install failed — see logs/web.log"
fi

# `next build` and `next dev` share frontend/.next, and a production build left in there
# breaks the dev server in a way that does not look like a build problem at all: the page
# loads, then the console fills with
#   WebSocket connection to 'ws://localhost:3000/_next/hmr?id=…' failed
# repeating forever, because hot reload is talking to a build id that no longer exists.
#
# `BUILD_ID` is written by `next build` and never by `next dev`, so its presence is an exact
# signal that this directory holds the wrong kind of build. Clearing it costs one cold
# recompile and removes a failure mode that otherwise needs somebody to know about it.
if [ -f "$ROOT/frontend/.next/BUILD_ID" ]; then
  info "frontend/.next holds a production build (npm run build) — clearing it for next dev"
  rm -rf "$ROOT/frontend/.next"
fi
start_web() {
  (cd "$ROOT/frontend" && npm run dev) >>"$WEB_LOG" 2>&1 &
  WEB_PID=$!
  if wait_for_port "$WEB_PORT" 240 "$WEB_PID"; then
    info "listening (log: logs/web.log)"
    return 0
  fi
  printf '\n--- last 25 lines of logs/web.log ---\n'
  tail -25 "$WEB_LOG"
  die "the UI did not open port $WEB_PORT."
}

reclaim_port "$WEB_PORT" web
start_web

# Compile /login now rather than on the first click. `next dev` builds a route on demand,
# so this is the difference between a sign-in page that appears instantly and one that
# spends 20 seconds blank.
info "warming up /login…"
login_status="$(http_status "http://localhost:$WEB_PORT/login" 150)"

# **A 404 here means the build cache, not a missing route.** `frontend/app/login/page.tsx`
# is in git; what goes wrong is `next dev` being force-killed mid-compile, which leaves a
# half-written `.next/` whose manifests no longer list the route. The page then 404s until
# the directory is cleared — and since stopping this script force-kills the dev server, the
# next run is where it surfaces. `.next/` is a cache, so deleting it is free.
if [ "$login_status" = "404" ]; then
  info "/login answered 404 — stale .next build cache. Clearing it and restarting the UI."
  reclaim_port "$WEB_PORT" web
  rm -rf "$ROOT/frontend/.next"
  start_web
  info "recompiling /login (a cold build, so this is slower)…"
  login_status="$(http_status "http://localhost:$WEB_PORT/login" 240)"
fi

case "$login_status" in
  200) info "compiled" ;;
  000) info "still compiling — the first page load will take a few seconds" ;;
  *) info "WARNING: /login answered HTTP $login_status — see logs/web.log" ;;
esac

# ----------------------------------------------------------------------------- ready

say "ready"
info "UI   http://localhost:$WEB_PORT"
info "API  http://$API_HOST:$API_PORT/docs"
info "Ctrl-C stops the API and the UI. PostgreSQL keeps running."
printf '\n-- following logs/api.log and logs/web.log --\n\n'

trap shutdown INT TERM
tail -n 0 -f "$API_LOG" "$WEB_LOG" &
TAIL_PID=$!
wait "$TAIL_PID"
shutdown
