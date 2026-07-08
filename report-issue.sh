#!/usr/bin/env bash
# =============================================================================
#  LocaLM - report a problem, even when localm will not start (Linux / macOS).
#
#  Run:  bash report-issue.sh
#  Files a bug report as an account-less GitHub issue via the localm proxy (no
#  GitHub login), showing you exactly what will be sent and asking you to confirm
#  first. It runs the self-contained reporter on any Python it can find (so a
#  broken venv still reports fine), and falls back to curl if there is no Python.
#
#  Optional args (used by setup.sh's failure paths) pass straight through:
#    --summary "..."  --detail "..."  --log <path>  --yes
# =============================================================================
cd "$(dirname "$0")" || exit 1

# Prefer the clone's venv python; else any python3/python. All run the SAME
# stdlib-only reporter (needs no working localm install).
RIPY=""
for c in ".venv/bin/python3" ".venv/bin/python"; do
  [ -x "$c" ] && { RIPY="$c"; break; }
done
if [ -z "$RIPY" ]; then
  for c in python3 python; do
    command -v "$c" >/dev/null 2>&1 && { RIPY="$c"; break; }
  done
fi

if [ -n "$RIPY" ]; then
  exec "$RIPY" scripts/report_issue.py "$@"
fi

# --- No Python at all: minimal curl reporter (a rare fallback) ---------------
echo ""
echo "  localm - report a problem (curl fallback; no Python found)"
echo ""
email="theilige@gmail.com"
if ! command -v curl >/dev/null 2>&1; then
  echo "  Cannot file automatically: no Python and no curl on this machine."
  echo "  Please email a short description to ${email}."
  exit 1
fi

cfg="localm/config.py"
url="${LOCALM_BUGREPORT_URL:-}"; tok="${LOCALM_BUGREPORT_TOKEN:-}"
if [ -z "$url" ] && [ -f "$cfg" ]; then
  url="$(sed -n 's/.*"bugreport_upload_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$cfg" | head -1)"
fi
if [ -z "$tok" ] && [ -f "$cfg" ]; then
  tok="$(sed -n 's/.*"bugreport_upload_token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$cfg" | head -1)"
fi

printf '  One line - what went wrong? '; IFS= read -r summary
[ -n "$summary" ] || summary="localm would not start"
os="$(uname -srm 2>/dev/null || echo unknown)"
# Single-line body so JSON escaping stays simple and safe (quotes + backslashes).
esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
body="Filed via the standalone curl reporter (no Python available). OS: ${os}. Details: ${summary}"

echo ""
echo "  ----- this is exactly what will be sent -----"
echo "  title: $summary"
echo "  body:  $body"
echo "  ---------------------------------------------"
printf '  Send this to the maintainer now? [Y/n]: '; IFS= read -r ans
case "$ans" in [Nn]*) echo "  Not sent. Email ${email} if you change your mind."; exit 0 ;; esac

if [ -z "$url" ]; then
  echo "  No bug-report endpoint is configured. Email ${email}."
  exit 1
fi

payload="{\"title\":\"$(esc "$summary")\",\"body\":\"$(esc "$body")\"}"
code="$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$url" \
  -H "Content-Type: application/json" ${tok:+-H "X-Localm-Token: $tok"} \
  -d "$payload" 2>/dev/null || echo 000)"
if [ "$code" = "201" ] || [ "$code" = "200" ]; then
  echo "  Sent to the maintainer. Thank you!"
else
  echo "  Could not send it (HTTP ${code}). Please email ${email} instead."
  exit 1
fi
