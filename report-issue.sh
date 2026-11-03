#!/usr/bin/env bash
# =============================================================================
#  LocaLM - report a problem, even when localm will not start (Linux / macOS).
#
#  Run:  bash report-issue.sh
#  Files a bug report as an account-less GitHub issue via the localm proxy (no
#  GitHub login), showing you exactly what will be sent and asking you to confirm
#  first. It runs the self-contained reporter on localm's own Python (so a
#  broken venv still reports fine), and falls back to the system curl (or wget)
#  if there is none.
#
#  Optional args (used by setup.sh's failure paths) pass straight through:
#    --summary "..."  --detail "..."  --log <path>  --yes
# =============================================================================
# Inherited uv / pip / Python settings and LOCALM_HOME are cleared, and PATH is
# the fixed system folders.
LOCALM_SYSDIRS="/usr/bin:/bin:/usr/sbin:/sbin"
export PATH="$LOCALM_SYSDIRS"
for _var in $(compgen -e); do
  case "$_var" in UV_*|PIP_*) unset "$_var" ;; esac
done
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX LOCALM_HOME
export UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1
case "$0" in */*) cd -- "${0%/*}" || exit 1 ;; esac

sys_prog() {  # sys_prog NAME  ->  prints NAME's full path in the fixed system folders; 1 when absent
  local d IFS=:
  for d in $LOCALM_SYSDIRS; do
    if [ -x "$d/$1" ] && [ ! -d "$d/$1" ]; then printf '%s\n' "$d/$1"; return 0; fi
  done
  return 1
}

# The reporter runs on this clone's .venv Python, else on localm's own managed
# Python (./.python, or the Shared store); never on a Python from PATH. All run
# the SAME stdlib-only reporter (needs no working localm install).
case "${XDG_DATA_HOME:-}" in
  /*) SHARED_STORE="$XDG_DATA_HOME/localm/shared" ;;
  *)  SHARED_STORE="${HOME:+$HOME/.local/share/localm/shared}" ;;
esac
RIPY=""
for c in .venv/bin/python3 .venv/bin/python .python/*/bin/python3 \
    "${SHARED_STORE:-/nonexistent}"/python/*/bin/python3; do
  [ -x "$c" ] && { RIPY="$c"; break; }
done

if [ -n "$RIPY" ]; then
  exec "$RIPY" scripts/report_issue.py "$@"
fi

# --- No Python at all: minimal curl / wget reporter (a rare fallback) --------
echo ""
echo "  localm - report a problem (no Python found)"
echo ""
email="theilige@gmail.com"
curl_bin=""; wget_bin=""
if ! curl_bin="$(sys_prog curl)" && ! wget_bin="$(sys_prog wget)"; then
  echo "  Cannot file automatically: no Python, curl or wget on this machine."
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

# Strip the account name, URL credentials, credential-named parameters, header
# lines and JSON keys, bearer tokens, API keys and email addresses other than
# RI_EMAIL from stdin, line by line. The same rules, in the same order, as
# _scrub_secrets in localm/bugreport/scrub.py. Returns non-zero, printing
# nothing, when perl is not available.
scrub() {
  local perl_bin
  perl_bin="$(sys_prog perl)" || return 1
  "$perl_bin" -CS -e "$(cat <<'PERL'
    use strict;
    use warnings;
    my $home = $ENV{HOME} // '';
    my $maint = $ENV{RI_EMAIL} // '';
    my $E = '\x00-\x20\x7f-\xa0\x22<>()\[\],;:/\\\\\x60=?&*';
    use Sys::Hostname;
    my @hn;
    { my $h = eval { hostname() }; push @hn, $h if defined $h; }
    push @hn, $ENV{COMPUTERNAME} if defined $ENV{COMPUTERNAME};
    my %names;
    for my $n (@hn) {
      $n =~ s/^\s+|\s+\z//g;
      $n =~ s/\.+\z//;
      next if $n eq '';
      $names{$n} = 1;
      (my $first = $n) =~ s/\..*//s;
      $names{$first} = 1;
    }
    my @keep = sort { length($b) <=> length($a) || $a cmp $b }
               grep { length($_) >= 3 && lc($_) ne 'localhost' } keys %names;
    my $own = join('|', map { quotemeta } grep { !/\A[A-Za-z]+\z/ } @keep);
    my $word = join('|', map { quotemeta } @keep);
    sub email_repl {
      my $tok = shift;
      (my $core = $tok) =~ s/^'+//;
      my $lead = substr($tok, 0, length($tok) - length($core));
      $core =~ s/['.]+\z//;
      my $trail = substr($tok, length($lead) + length($core));
      (my $addr = $core) =~ s/%40/\@/g;
      return $tok if $maint ne '' && $addr !~ /[^\x00-\x7f]/ && lc($addr) eq lc($maint);
      return $lead . '<redacted-email>' . $trail;
    }
    while (my $t = <STDIN>) {
      my $nl = ($t =~ s/\n\z//) ? "\n" : '';
      if ($home ne '') {
        (my $fwd = $home) =~ tr{\\}{/};
        (my $dbl = $home) =~ s{\\}{\\\\}g;
        for my $v ($home, $fwd, $dbl) { $t =~ s{\Q$v\E}{~}g; }
      }
      $t =~ s{([A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)|/home/|/Users/)[^\\/\r\n]+}{$1<redacted>}g;
      $t =~ s{(://)[^/@\s]+\@}{$1<redacted>\@}g;
      $t =~ s{(?i)((?:(?<=[?&])(?:api[_-]?key|key|token|secret|password|passwd|pwd|auth|access[_-]?token|sig|signature)|(?<![A-Za-z0-9])(?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature)|(?<=[A-Za-z0-9])[_-](?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature|key|auth|sig))=)(?!["']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])[`"'\)\]\}]{0,4}(?:[\s&#]|$))(?:"[^"\r\n]*"?|'[^'\r\n]*'?|[^&\s#"'\)\]\}]*)}{$1<redacted>}g;
      $t =~ s{(?i)((?:x-)?(?:api[_-]key|api[_-]token|auth[_-]token|authorization)\s*:\s*)(?:(?:bearer|basic|digest|negotiate|ntlm)\s+)?\S+}{$1<redacted>}g;
      $t =~ s{(?i)(["'](?:(?:[A-Za-z0-9]+[_-])?(?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature)|[A-Za-z0-9]+[_-](?:key|auth|sig))["']\s*:\s*)(?!\s*["']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])["']?\s*(?:[,\}\]]|$))(?:"[^"\r\n]*"?|'[^'\r\n]*'?|[^,\}\]\s]*)}{$1<redacted>}g;
      $t =~ s{(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}}{$1<redacted>}g;
      $t =~ s{(?i)\b(?:sk|localm[_-]sk)-[A-Za-z0-9._\-]{12,}}{<redacted>}g;
      $t =~ s{(?<![^$E])[^$E]*?[^\@$E](?:\@|%40)[^\@.%+$E]+(?:\.[^\@.%+$E]+)*\.[^\x00-\x40\x5b-\x60\x7b-\xa0]{2,}[^$E]*}{email_repl($&)}ge;
      $t =~ s~(?:(?<![^\s\x22\x27(\[{=,;|\x60<*])|(?<=[A-Za-z0-9_]{2}:))(\\{1,2}\\{1,2}(?:\?\\{1,2}UNC\\{1,2})?)(?!(?:wsl\$|wsl\.localhost|localhost)(?![A-Za-z0-9._\$-]))[A-Za-z0-9][A-Za-z0-9._\$-]*~$1<host>~gi;
      $t =~ s~(?<![A-Za-z0-9._-])(?!(?:localm\.local|(?:host|gateway|kubernetes)\.docker\.internal)(?![A-Za-z0-9-]|\.[A-Za-z0-9]))[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(local|lan|home\.arpa|internal|localdomain)(?![A-Za-z0-9(-]|\.[A-Za-z0-9])~<host>.$1~gi;
      $t =~ s~(?<![A-Za-z0-9_./\\-])(?:$own)(?![A-Za-z0-9_-]|[./\\][A-Za-z0-9_])~<host>~gi if $own ne '';
      if ($word ne '') {
        $t =~ s~(?:(?<=//)|(?<=\@))(?:$word)(?![A-Za-z0-9_-]|\.[A-Za-z0-9])~<host>~gi;
        $t =~ s~(?<![A-Za-z0-9_./\\\@<-])(?:$word)(?=:\d)~<host>~gi;
        $t =~ s~(?<![A-Za-z0-9_])((?:host|computer|machine|node|device|pc)(?:[ _-]?name)?["']?[ \t]*[:=][ \t]*["']?)(?:$word)(?![A-Za-z0-9_-]|\.[A-Za-z0-9])~$1<host>~gi;
      }
      print $t . $nl;
    }
PERL
)"
}

# Nothing is sent without an answer typed at a terminal.
if [ ! -t 0 ]; then
  echo "  No terminal to confirm on, so nothing was sent. Email ${email} instead."
  exit 0
fi

printf '  One line - what went wrong? '; IFS= read -r summary || summary=""
[ -n "$summary" ] || summary="localm would not start"
if ! summary="$(printf '%s\n' "$summary" | RI_EMAIL="$email" scrub)" || [ -z "$summary" ]; then
  echo "  Cannot remove personal details from the report here (perl is not installed),"
  echo "  so nothing was sent. Please email a short description to ${email}."
  exit 1
fi
os="$(uname -srm 2>/dev/null || echo unknown)"
# Single-line body so JSON escaping stays simple and safe (quotes + backslashes).
esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
body="Filed via the standalone shell reporter (no Python available). OS: ${os}. Details: ${summary}"

echo ""
echo "  ----- this is exactly what will be sent -----"
echo "  title: $summary"
echo "  body:  $body"
echo "  ---------------------------------------------"
printf '  Send this to the maintainer now? [Y/n]: '
if ! IFS= read -r ans; then
  echo ""
  echo "  Not sent. Email ${email} if you change your mind."
  exit 0
fi
case "$ans" in [Nn]*) echo "  Not sent. Email ${email} if you change your mind."; exit 0 ;; esac

if [ -z "$url" ]; then
  echo "  No bug-report endpoint is configured. Email ${email}."
  exit 1
fi

payload="{\"title\":\"$(esc "$summary")\",\"body\":\"$(esc "$body")\"}"
if [ -n "$curl_bin" ]; then
  code="$("$curl_bin" -sS -o /dev/null -w '%{http_code}' -X POST "$url" \
    -H "Content-Type: application/json" ${tok:+-H "X-Localm-Token: $tok"} \
    -d "$payload" 2>/dev/null || echo 000)"
else
  # wget prints the response headers on stderr with --server-response; the last
  # status line is the answer.
  code="$("$wget_bin" -q -O /dev/null --server-response \
    --header="Content-Type: application/json" ${tok:+--header="X-Localm-Token: $tok"} \
    --post-data="$payload" "$url" 2>&1 | sed -n 's/^ *HTTP\/[0-9.]* \([0-9][0-9][0-9]\).*/\1/p' | tail -1)"
  code="${code:-000}"
fi
if [ "$code" = "201" ] || [ "$code" = "200" ]; then
  echo "  Sent to the maintainer. Thank you!"
else
  echo "  Could not send it (HTTP ${code}). Please email ${email} instead."
  exit 1
fi
