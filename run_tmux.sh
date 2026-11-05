#!/usr/bin/env bash
#
# run_tmux.sh — run the DocQuery stack on the POC instance inside one tmux session.
#
# Three panes in a single window, so one `attach` shows the whole stack at once:
#
#     +---------------------------+---------------------------+
#     |  backend  127.0.0.1:9077  |  frontend 127.0.0.1:3300  |
#     |  .venv + scripts/dev_...  |  npm run start            |
#     +---------------------------+---------------------------+
#     |  edge proxy  0.0.0.0:8877  <- the only public port    |
#     |  deploy/edge_proxy.py — one line per request          |
#     +-------------------------------------------------------+
#
# **Why a proxy.** The instance's firewall lets exactly one port through to this account,
# and the browser needs two services: the UI, and the API the UI calls cross-origin. So
# both servers bind loopback and `deploy/edge_proxy.py` fronts them on 8877, routing
# `/api/…`, `/health`, `/readiness` and `/docs` to the API and everything else to Next.
# No application code changes; the split is entirely in this directory and in deploy/.
# `EDGE=0 API_HOST=0.0.0.0 WEB_HOST=0.0.0.0 ./run_tmux.sh` goes back to two public ports
# for when the firewall is opened.
#
#     ./run_tmux.sh              # start (or attach to) the session, then attach
#     ./run_tmux.sh --attach     # attach to a session that is already running
#     ./run_tmux.sh --no-attach  # start it detached (what you want over a dropping SSH link)
#     ./run_tmux.sh --stop       # kill the session; PostgreSQL is left running
#     ./run_tmux.sh --status     # is it up, and what is listening
#     ./run_tmux.sh --no-build   # skip `next build` and serve the existing frontend/.next
#                                # (only when the code has not changed since it was built)
#
# Why tmux and not `start.sh`: `start.sh` is the Windows developer script (netstat,
# taskkill, pg_ctl.exe) and it dies with the terminal. On the POC instance the stack has
# to outlive the SSH session, so each process gets a tmux pane and the session keeps
# running after you detach with `Ctrl-b d`.
#
# Everything is overridable from the environment, so nothing in here is machine-specific:
#
#     SESSION=docquery EDGE_PORT=8877 API_PORT=9077 WEB_PORT=3300 \
#     WEB_MODE=dev API_RELOAD=1 EDGE=1 ./run_tmux.sh
#
# All three ports are reclaimed before binding, and again on --stop. A `next-server`
# orphaned by an earlier run keeps its socket after its tmux pane is gone, and the only
# symptom is `EADDRINUSE` at the end of the next build — a busy port that reads like a
# permissions fault. Only a listener this user owns, whose process looks like part of this
# stack, is ever killed; anything else stops the script with its name.
#
# No credential is read, printed or passed on a command line. `.env` is checked for the
# *presence* of a few keys and never for their values (`.claude/rules/secure-coding-vapt.md`).
#
# This script starts processes only. It changes no application code and no schema beyond
# `alembic upgrade head` (skip that with --no-migrate).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

SESSION="${SESSION:-docquery}"
# The only port the instance's firewall lets through to this account. Everything the
# browser touches arrives here and is routed by deploy/edge_proxy.py.
EDGE_HOST="${EDGE_HOST:-0.0.0.0}"
EDGE_PORT="${EDGE_PORT:-8877}"
# Both application servers are loopback-only: the proxy is the single reachable surface,
# so a direct hit on the API or on Next from the network is impossible rather than merely
# unrouted. Set EDGE=0 to bind them publicly and skip the proxy instead.
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-9077}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-3300}"
EDGE="${EDGE:-1}"
# dev = `next dev` (compiles on demand, slower first paint, no build step)
# prod = `next build` once, then `next start` (what a demo should use)
# `prod` — `next build` + `next start` — because this script exists to run the stack on the
# shared instance behind `deploy/edge_proxy.py`, and `next dev` is wrong there in two ways that
# both show up as a console full of red on a page that otherwise works:
#
#   * **Hot reload cannot work through the edge proxy.** `next dev` opens a WebSocket to
#     `/_next/hmr`, and the proxy is a stdlib HTTP server with no `101 Switching Protocols`
#     handling — its own module docstring says so. The socket therefore fails, forever, once
#     per retry.
#   * **Dev chunk URLs are not stable.** `next dev` compiles on demand and renames chunks as it
#     goes, so a browser holding a page from before the last compile asks for
#     `/_next/static/chunks/<old>` and gets a 4xx. Behind a TLS-terminating proxy that arrives
#     as `net::ERR_ABORTED 403`, which looks like a permissions problem and is not one.
#
# A production build has neither property: no HMR socket at all, and content-hashed filenames
# that only change when the content does. It is also what the public URL should be serving.
#
# `WEB_MODE=dev ./run_tmux.sh` still gives the old behaviour for someone editing on the box.
WEB_MODE="${WEB_MODE:-prod}"

# Rebuild on start in `prod` mode. `--no-build` skips it.
#
# The build used to be skipped whenever `.next/BUILD_ID` existed, which is the file `next build`
# writes — so after `deploy/push.sh` delivered new code, the restart found the *previous*
# build's marker and served the old bundle. A deploy that changes nothing visible is the worst
# possible failure mode: it looks like the push failed, so the next instinct is to push again.
WEB_BUILD="${WEB_BUILD:-1}"
# uvicorn --reload watches .py files. Off by default here: a POC instance is not edited
# in place, and the reloader doubles the process count in the pane.
API_RELOAD="${API_RELOAD:-0}"
VENV="${VENV:-$ROOT/.venv}"
LOG_DIR="$ROOT/logs"
# Resolved settings for the panes to read. tmux's own -e needs tmux >= 3.2 and the session
# environment is inherited only at creation time, so a file is the portable way to hand
# the same values to a pane. Hosts and ports only — never a secret.
PANE_ENV="$LOG_DIR/run_tmux.env"

say() { printf '\n== %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------- virtualenv layout

# Linux puts the venv's programs in bin/, Windows in Scripts/. Resolve once, here, so the
# rest of the script and both panes agree.
if [ -f "$VENV/bin/activate" ]; then
  VENV_BIN="$VENV/bin"
elif [ -f "$VENV/Scripts/activate" ]; then
  VENV_BIN="$VENV/Scripts"
else
  VENV_BIN=""
fi

venv_python() {
  if [ -x "$VENV_BIN/python" ]; then
    printf '%s' "$VENV_BIN/python"
  else
    printf '%s' "$VENV_BIN/python.exe"
  fi
}

# ------------------------------------------------------------------------------ the panes
#
# Each pane re-enters this script with --pane. That keeps the two command lines in this
# file instead of in a tmux string that would have to survive two levels of quoting, and
# it means `bash -lc` runs them: a login shell, so a node installed by nvm or by a
# profile script is on PATH exactly as it is for an interactive login.

run_pane() {
  # shellcheck disable=SC1090
  [ -f "$PANE_ENV" ] && . "$PANE_ENV"
  cd "$ROOT" || exit 1

  case "$1" in
    backend)
      # shellcheck disable=SC1091
      . "$VENV_BIN/activate" || exit 1
      printf '=== backend  http://%s:%s/docs  (.venv active: %s)\n\n' \
        "$API_HOST" "$API_PORT" "$VIRTUAL_ENV"
      reload_flag="--no-reload"
      [ "$API_RELOAD" = "1" ] && reload_flag=""
      # dev_server.py, not bare uvicorn: it points TLS at .certs/corporate-ca-bundle.pem
      # and strips the AWS_*/PINECONE_* shell variables that would otherwise shadow .env.
      # shellcheck disable=SC2086
      exec python scripts/dev_server.py --host "$API_HOST" --port "$API_PORT" $reload_flag
      ;;
    frontend)
      cd "$ROOT/frontend" || exit 1
      printf '=== frontend  http://%s:%s  (mode: %s)\n\n' "$WEB_HOST" "$WEB_PORT" "$WEB_MODE"
      if [ ! -d node_modules ]; then
        printf -- '--- node_modules missing, installing (first run only)\n'
        npm install || exit 1
      fi
      if [ "$WEB_MODE" = "prod" ]; then
        # Built unless told not to, and unconditionally when there is no build at all. The
        # old `[ ! -f .next/BUILD_ID ]` test alone meant a push followed by a restart served
        # the previous bundle — see `WEB_BUILD`.
        if [ "$WEB_BUILD" = "1" ] || [ ! -f .next/BUILD_ID ]; then
          printf -- '--- next build\n'
          npm run build || exit 1
        else
          printf -- '--- skipping next build (WEB_BUILD=0) — serving the existing .next\n'
        fi
        exec npm run start -- --hostname "$WEB_HOST" --port "$WEB_PORT"
      fi
      # `next dev` keeps a `.next` with no BUILD_ID, so a later prod run rebuilds anyway.
      exec npm run dev -- --hostname "$WEB_HOST" --port "$WEB_PORT"
      ;;
    edge)
      # shellcheck disable=SC1091
      . "$VENV_BIN/activate" || exit 1
      # Stdlib only, so the venv is incidental — it is used for a predictable interpreter
      # version, not for a dependency.
      exec python deploy/edge_proxy.py \
        --listen "$EDGE_HOST:$EDGE_PORT" \
        --api "127.0.0.1:$API_PORT" \
        --web "127.0.0.1:$WEB_PORT"
      ;;
    *) die "unknown pane: $1" ;;
  esac
}

# ---------------------------------------------------------------------------- subcommands

session_exists() { tmux has-session -t "=$SESSION" 2>/dev/null; }

listening_on() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | grep -q ":$1 "
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | grep -q ":$1 "
  else
    return 2 # unknown, not "down"
  fi
}

port_report() {
  local port="$1" label="$2"
  listening_on "$port"
  case $? in
    0) info "$label  port $port — listening" ;;
    1) info "$label  port $port — nothing listening yet" ;;
    *) info "$label  port $port — cannot tell (no ss/netstat)" ;;
  esac
}

# --------------------------------------------------------------------------- stale ports
#
# `tmux kill-session` signals the pane's own process. `npm run start` forks `next start`,
# which forks `next-server`, and that grandchild is reparented to init and keeps the
# listening socket. The next start then dies with `EADDRINUSE` after a full rebuild — the
# port is busy, which looks nothing like the error it prints. Same story for uvicorn's
# reload supervisor. So the port is cleared before binding, and again after --stop.

# PIDs listening on a TCP port. `ss -p` only reveals the PID for sockets this user owns,
# which is exactly the set we are allowed to kill anyway.
pids_on_port() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null |
      awk -v port=":$1" '$4 ~ port"$"' |
      grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u
  elif command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null | sort -u
  fi
}

# Free a port, but only from a process that plausibly belongs to this stack. Killing an
# unknown listener silently is how a script eats something somebody else needed — on a
# shared instance that matters more than it does on a laptop.
reclaim_port() {
  local port="$1" label="$2" pid comm waited=0
  # Half-seconds to wait after SIGTERM before escalating to SIGKILL. Short by default, and
  # long for the API — `drain_ingestion_pool` in the lifespan waits up to 30s for the
  # documents it is parsing and embedding right now, and killing it at 3s would abandon that
  # work and leave those rows stuck at `PROCESSING`. It costs nothing in the normal case: the
  # loop below exits the moment the port is free, which for an idle API is immediate.
  local escalate_after="${3:-6}" give_up_after="${4:-10}"
  for pid in $(pids_on_port "$port"); do
    comm="$(ps -p "$pid" -o comm= 2>/dev/null | tr -d ' ')"
    case "$comm" in
      node | npm | next* | python | python3 | python3.* | uvicorn | sh | bash)
        info "port $port held by $comm (PID $pid) — stopping it ($label)"
        kill "$pid" 2>/dev/null
        ;;
      '')
        info "port $port held by PID $pid, which is not ours to inspect — leaving it alone"
        ;;
      *)
        die "port $port is held by $comm (PID $pid), which is not part of this stack.
       Stop it yourself, or start with ${label^^}_PORT set to a free port."
        ;;
    esac
  done
  # A graceful kill is enough for node and uvicorn; escalate only if the socket lingers.
  while [ -n "$(pids_on_port "$port")" ] && [ "$waited" -lt "$give_up_after" ]; do
    sleep 0.5
    waited=$((waited + 1))
    if [ "$waited" = "$escalate_after" ]; then
      info "port $port still held after $((waited / 2))s — escalating to SIGKILL"
      for pid in $(pids_on_port "$port"); do kill -9 "$pid" 2>/dev/null; done
    fi
  done
}

# Half-seconds of patience for the API port on --stop: 35s before SIGKILL, 40s before giving
# up. Sized to outlast the 30s ingestion drain plus the Pinecone and database pool teardown
# after it. Startup uses the short default instead — a listener found there is a stale one,
# not our own process finishing work.
API_ESCALATE_AFTER=70
API_GIVE_UP_AFTER=80

# Reclaiming is not enough on a shared host. `ss -p` prints no PID for a socket owned by
# another account, so such a listener is invisible to `pids_on_port` and survives the loop
# above — and the pane then dies with `EADDRINUSE` at the end of a full rebuild, which
# reads like a build failure. Checking the port is still *free* turns that into one line,
# before anything is built.
require_port_free() {
  local port="$1" label="$2" upper
  reclaim_port "$port" "$label"
  if listening_on "$port"; then
    upper="$(printf '%s' "$label" | tr '[:lower:]' '[:upper:]')"
    die "port $port is in use by a process this account cannot stop — on a shared host
       that means it belongs to another user (ss shows no PID for their socket).
       Pick a free port instead:
         ${upper}_PORT=<free port> ./run_tmux.sh
       Find one with:
         for p in 9082 9092 9095 9099 9190 9290; do ss -ltn | grep -q \":\$p \" ||
           echo \"\$p free\"; done
       If it is the UI port, also update FRONTEND_ORIGIN and CORS_ALLOW_ORIGINS in .env
       to match — the browser's origin has to be allowed by the API or every call fails
       CORS with a working API behind it."
  fi
}

do_status() {
  say "status"
  if session_exists; then
    info "tmux session '$SESSION' is running:"
    tmux list-panes -t "=$SESSION" -F '     pane #{pane_index} #{pane_title} — #{pane_current_command} (dead: #{pane_dead})'
  else
    info "no tmux session '$SESSION'"
  fi
  port_report "$API_PORT" "backend "
  port_report "$WEB_PORT" "frontend"
  # An `if`, not `[ … ] && …`: that form makes the false case the function's exit status,
  # and this is the last call in the non-attach path — the whole script would exit 1 with
  # EDGE=0 and nothing wrong.
  if [ "$EDGE" = "1" ]; then
    port_report "$EDGE_PORT" "edge    "
  fi
}

do_stop() {
  say "stopping"
  if session_exists; then
    tmux kill-session -t "=$SESSION"
    info "killed tmux session '$SESSION' — API and UI are down"
  else
    info "no tmux session '$SESSION' to kill"
  fi
  # The grandchildren outlive the panes, so killing the session is not the whole job.
  reclaim_port "$API_PORT" api "$API_ESCALATE_AFTER" "$API_GIVE_UP_AFTER"
  reclaim_port "$WEB_PORT" web
  if [ "$EDGE" = "1" ]; then
    reclaim_port "$EDGE_PORT" edge
  fi
  info "PostgreSQL is left running (it is a separate service here)"
}

do_attach() {
  session_exists || die "no tmux session '$SESSION' — start it with: ./run_tmux.sh"
  if [ -n "${TMUX:-}" ]; then
    tmux switch-client -t "=$SESSION"
  else
    tmux attach-session -t "=$SESSION"
  fi
}

usage() {
  sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

ATTACH=1
MIGRATE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --pane)
      [ $# -ge 2 ] || die "--pane needs a name"
      run_pane "$2"
      exit $?
      ;;
    --stop | stop)
      do_stop
      exit 0
      ;;
    --status | status)
      do_status
      exit 0
      ;;
    --attach | attach)
      do_attach
      exit $?
      ;;
    --no-attach | -d) ATTACH=0 ;;
    --no-migrate) MIGRATE=0 ;;
    # Serve whatever is already in frontend/.next. Only safe when the code has not changed
    # since it was built — a restart after `deploy/push.sh` must not use this.
    --no-build) WEB_BUILD=0 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done

# ------------------------------------------------------------------------------ preflight

say "preflight"
command -v tmux >/dev/null 2>&1 ||
  die "tmux is not installed — see deploy/README.md §2 (sudo dnf install tmux / sudo apt install tmux)"
case "$ROOT" in
  *\'*) die "the project path contains a single quote, which the pane commands cannot quote safely: $ROOT" ;;
esac
[ -n "$VENV_BIN" ] ||
  die "no virtualenv at $VENV — run: uv venv && uv pip install -r requirements.txt
       (or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt)"
[ -f "$ROOT/.env" ] ||
  die "no .env — copy deploy/env.poc.example to .env and fill in the real values"
if [ "$EDGE" = "1" ]; then
  [ -f "$ROOT/deploy/edge_proxy.py" ] ||
    die "EDGE=1 but deploy/edge_proxy.py is missing — redeploy with ./deploy/push.sh --dirty,
       or run without the proxy: EDGE=0 API_HOST=0.0.0.0 WEB_HOST=0.0.0.0 ./run_tmux.sh"
fi
mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"
# A warning, not a failure: the panes run through `bash -l`, so a node installed by nvm or
# by a profile script is on their PATH even when it is absent from this shell — which is
# exactly the case for a non-login `ssh host './run_tmux.sh'`.
if command -v node >/dev/null 2>&1; then
  info "tmux $(tmux -V | awk '{print $2}'), node $(node --version), venv $VENV_BIN"
else
  info "tmux $(tmux -V | awk '{print $2}'), venv $VENV_BIN"
  info "NOTE: node is not on PATH in this shell. The frontend pane uses a login shell, so"
  info "      an nvm-managed Node still works — if pane 1 dies, that is the first thing to check."
fi

# Presence only. A missing credential is reported without putting its value in the
# terminal, in a log, or in the process list.
env_has_value() {
  grep -Eq "^[[:space:]]*$1[[:space:]]*=[[:space:]]*[^[:space:]#]" "$ROOT/.env"
}
# Keys with no fallback anywhere: absent means the feature cannot work.
missing=""
for key in SECRET_KEY DATABASE_URL PINECONE_API_KEY PINECONE_INDEX_NAME S3_BUCKET_NAME; do
  env_has_value "$key" || missing="$missing $key"
done
if [ -n "$missing" ]; then
  info "WARNING: empty or absent in .env:$missing"
  info "the API will still boot; upload, embedding and retrieval fail until these are set"
else
  info ".env carries values for every required key (values not read)"
fi

# The AWS keys are **deliberately optional**, and reporting them with the list above was
# wrong: it stated "upload, embedding and retrieval fail until these are set", which is false
# on this instance. `Settings.bedrock_credentials()` returns `{}` when they are blank — see
# its docstring — which hands boto3 its ordinary credential chain, and on EC2 that chain
# includes the instance profile. Flagging a working configuration as broken is how an hour
# gets spent on a non-problem.
aws_missing=""
for key in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
  env_has_value "$key" || aws_missing="$aws_missing $key"
done
if [ -n "$aws_missing" ]; then
  info "note: no AWS keys in .env ($aws_missing ) — boto3 will use its own credential chain"
  info "expected on an instance with a role attached; set them only if AWS returns AccessDenied"
fi

# cookies_secure is derived as ENVIRONMENT != "local", so it is coupled to the scheme the
# browser actually uses (deploy/README.md §6). Both mismatches fail silently, in opposite
# directions, so warn on either — and say nothing when the two agree. This used to warn
# unconditionally whenever ENVIRONMENT was not 'local', which told a correctly configured
# HTTPS instance to revert the one setting it needed.
env_value() { sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$ROOT/.env" | tail -n 1; }
FRONTEND_ORIGIN_VALUE="$(env_value FRONTEND_ORIGIN)"
if env_has_value ENVIRONMENT; then
  if grep -Eq '^[[:space:]]*ENVIRONMENT[[:space:]]*=[[:space:]]*local' "$ROOT/.env"; then
    case "${FRONTEND_ORIGIN_VALUE:-}" in
      https://*)
        info "WARNING: ENVIRONMENT=local but FRONTEND_ORIGIN is https — session cookies ship"
        info "without Secure, and /docs plus /openapi.json are served publicly."
        info "Set ENVIRONMENT=dev and restart — deploy/README.md §6."
        ;;
    esac
  else
    case "${FRONTEND_ORIGIN_VALUE:-}" in
      http://*)
        info "WARNING: ENVIRONMENT is not 'local' but FRONTEND_ORIGIN is http — the browser"
        info "discards the Secure cookies, so login returns 200 and every later call is 401."
        info "Serve over HTTPS or set ENVIRONMENT=local — deploy/README.md §6."
        ;;
    esac
  fi
fi

if session_exists; then
  info "tmux session '$SESSION' already exists — not starting a second one"
  if [ "$ATTACH" = "1" ] && [ -t 1 ]; then
    do_attach
    exit $?
  fi
  do_status
  exit 0
fi

# ----------------------------------------------------------------------------- migrations

if [ "$MIGRATE" = "1" ]; then
  say "schema"
  if [ -x "$VENV_BIN/alembic" ]; then
    ALEMBIC=("$VENV_BIN/alembic")
  elif [ -x "$VENV_BIN/alembic.exe" ]; then
    ALEMBIC=("$VENV_BIN/alembic.exe")
  else
    ALEMBIC=("$(venv_python)" -m alembic)
  fi
  if "${ALEMBIC[@]}" upgrade head >>"$LOG_DIR/alembic.log" 2>&1; then
    info "alembic upgrade head — at head"
  else
    printf '\n--- last 20 lines of logs/alembic.log ---\n'
    tail -20 "$LOG_DIR/alembic.log"
    die "alembic upgrade head failed. PostgreSQL down, or DATABASE_URL wrong?
       Start it, check .env, then run again — or skip with --no-migrate."
  fi
else
  info "--no-migrate: skipping alembic"
fi

# -------------------------------------------------------------------------------- session

say "ports"
require_port_free "$API_PORT" api
require_port_free "$WEB_PORT" web
if [ "$EDGE" = "1" ]; then
  require_port_free "$EDGE_PORT" edge
fi
if [ "$EDGE" = "1" ]; then
  info "edge $EDGE_PORT (public), api $API_PORT and web $WEB_PORT (loopback) are free"
else
  info "backend $API_PORT and frontend $WEB_PORT are free (EDGE=0, no proxy)"
fi

say "starting tmux session '$SESSION'"

# What the panes read. Rewritten every start so an override on the command line is what
# the panes actually use.
cat >"$PANE_ENV" <<EOF
# Written by run_tmux.sh — resolved settings for the panes. No secrets here.
API_HOST='$API_HOST'
API_PORT='$API_PORT'
WEB_HOST='$WEB_HOST'
WEB_PORT='$WEB_PORT'
WEB_MODE='$WEB_MODE'
WEB_BUILD='$WEB_BUILD'
API_RELOAD='$API_RELOAD'
EDGE_HOST='$EDGE_HOST'
EDGE_PORT='$EDGE_PORT'
EDGE='$EDGE'
VENV_BIN='$VENV_BIN'
EOF

pane_cmd() { printf "bash -lc '\"%s/run_tmux.sh\" --pane %s'" "$ROOT" "$1"; }

tmux new-session -d -s "$SESSION" -n stack -c "$ROOT" "$(pane_cmd backend)" ||
  die "tmux could not create the session"

# A pane that exits stays visible instead of vanishing, so a crash leaves its traceback on
# screen. Respawn one with: tmux respawn-pane -k -t docquery:stack.0
# `-w`: remain-on-exit and pane-border-status are window options, and `set-option` without
# it looks them up among the session options and fails. Set before the split so a process
# that dies in its first second still leaves its output behind.
tmux set-option -w -t "=$SESSION:stack" remain-on-exit on >/dev/null
tmux set-option -w -t "=$SESSION:stack" pane-border-status top >/dev/null 2>&1
tmux set-option -t "=$SESSION" mouse on >/dev/null 2>&1

tmux split-window -h -t "=$SESSION:stack" -c "$ROOT" "$(pane_cmd frontend)" ||
  die "tmux could not split the window"
tmux select-layout -t "=$SESSION:stack" even-horizontal >/dev/null

# The proxy gets the full width underneath the two servers it fronts: its log is one line
# per request, so it reads as a request trace while the two app logs sit above it.
# `-f` makes the new pane span the window rather than splitting only the pane it targets.
if [ "$EDGE" = "1" ]; then
  tmux split-window -v -f -l 30% -t "=$SESSION:stack" -c "$ROOT" "$(pane_cmd edge)" ||
    die "tmux could not add the proxy pane"
fi

# Pane titles need tmux >= 2.6, the border status that displays them >= 2.3. Both are
# cosmetic, so an older tmux just gets untitled panes rather than an error.
tmux select-pane -t "=$SESSION:stack.0" -T " backend :$API_PORT " >/dev/null 2>&1
tmux select-pane -t "=$SESSION:stack.1" -T " frontend :$WEB_PORT " >/dev/null 2>&1

# Mirror each pane to a file as well as the screen, so output that has scrolled past the
# end of tmux's buffer is still diagnosable after the fact (and greppable, which a
# scrollback is not). -o leaves an existing pipe alone instead of doubling it.
tmux pipe-pane -o -t "=$SESSION:stack.0" "cat >> '$LOG_DIR/api.log'" 2>/dev/null
tmux pipe-pane -o -t "=$SESSION:stack.1" "cat >> '$LOG_DIR/web.log'" 2>/dev/null
if [ "$EDGE" = "1" ]; then
  tmux select-pane -t "=$SESSION:stack.2" -T " edge :$EDGE_PORT (public) " >/dev/null 2>&1
  tmux pipe-pane -o -t "=$SESSION:stack.2" "cat >> '$LOG_DIR/edge.log'" 2>/dev/null
fi

tmux select-pane -t "=$SESSION:stack.0" >/dev/null

info "pane 0  backend   127.0.0.1:$API_PORT      (log: logs/api.log)"
info "pane 1  frontend  127.0.0.1:$WEB_PORT      (log: logs/web.log)"
if [ "$EDGE" = "1" ]; then
  info "pane 2  edge      http://$EDGE_HOST:$EDGE_PORT  <- the only public port (log: logs/edge.log)"
  # The public entry point is whatever FRONTEND_ORIGIN says, not a hardcoded IP. With Secure
  # cookies on (ENVIRONMENT != local) the raw http:// address cannot sign anyone in — the
  # browser discards the cookies — so printing it as "open this" sent people to the one URL
  # guaranteed to look broken.
  info "open    ${FRONTEND_ORIGIN_VALUE:-http://$EDGE_HOST:$EDGE_PORT}      (the public entry point)"
  # /docs follows ENABLE_API_DOCS when that is set, and ENVIRONMENT otherwise — the same
  # precedence as `Settings.api_docs_enabled`. Reporting it from ENVIRONMENT alone would now
  # tell an instance with the docs explicitly enabled that they are off.
  DOCS_SETTING="$(env_value ENABLE_API_DOCS | tr '[:upper:]' '[:lower:]')"
  case "${DOCS_SETTING:-}" in
    true | 1 | yes | on) DOCS_ON=1 ;;
    false | 0 | no | off) DOCS_ON=0 ;;
    *) grep -Eq '^[[:space:]]*ENVIRONMENT[[:space:]]*=[[:space:]]*local' "$ROOT/.env" &&
      DOCS_ON=1 || DOCS_ON=0 ;;
  esac
  if [ "$DOCS_ON" = "1" ]; then
    info "API docs at /docs on the same port"
  else
    info "note: /docs is off (set ENABLE_API_DOCS=true in .env to serve it)"
  fi
  if grep -Eq '^[[:space:]]*ENVIRONMENT[[:space:]]*=[[:space:]]*local' "$ROOT/.env"; then
    :
  else
    info "note: http://$EDGE_HOST:$EDGE_PORT cannot sign in — Secure cookies need the"
    info "      HTTPS origin above"
  fi
fi
info "detach with Ctrl-b d — everything keeps running"
info "come back with: ./run_tmux.sh --attach      stop with: ./run_tmux.sh --stop"

if [ "$ATTACH" = "1" ] && [ -t 1 ]; then
  do_attach
else
  say "running detached"
  if [ "$WEB_MODE" = "prod" ] && [ "$WEB_BUILD" = "1" ]; then
    info "the frontend runs 'next build' first — allow a minute before the UI answers"
  else
    info "the frontend needs a moment on a cold start ($WEB_MODE mode compiles on demand)"
  fi
  do_status
fi
