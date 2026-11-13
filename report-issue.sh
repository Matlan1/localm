#!/usr/bin/env bash
# =============================================================================
#  LocaLM - report a problem, even when localm will not start (Linux / macOS).
#
#  Run:  bash report-issue.sh
#  Files a bug report as an account-less GitHub issue via the localm proxy (no
#  GitHub login), showing you exactly what will be sent and asking you to confirm
#  first. It runs the self-contained reporter on any Python it can find (so a
#  broken venv still reports fine), and falls back to curl if there is no Python.
#  With localm's Network access off (LOCALM_NET_MODE, or net_mode in the data dir's
#  config.json) nothing is sent; when that setting cannot be read you are asked
#  before anything is sent.
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

# Strip the account name, URL credentials, credential-named parameters, header
# lines and JSON keys, bearer tokens, API keys and email addresses other than
# RI_EMAIL from stdin, line by line. The same rules, in the same order, as
# _scrub_secrets in localm/bugreport/scrub.py. Returns non-zero, printing
# nothing, when perl is not available.
scrub() {
  command -v perl >/dev/null 2>&1 || return 1
  perl -CS -e "$(cat <<'PERL'
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
      $t =~ s{(?i)(bearer\s+)(?![A-Za-z0-9_-]+(?:\s*:|[ \t]*=[ \t]*[^=\s]))[A-Za-z0-9._~+/=\-]{8,}}{$1<redacted>}g;
      $t =~ s{(?i)((?:(?<=[?&])(?:api[_-]?key|key|token|secret|password|passwd|pwd|auth|access[_-]?token|sig|signature)|(?<![A-Za-z0-9])(?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature)|(?<=[A-Za-z0-9])[_-](?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature|key|auth|sig))[ \t]*=[ \t]*(?![ \t]))(?!["']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])[`"'\)\]\}]{0,4}(?:[\s&#]|$))(?:"[^"\r\n]*"?|'[^'\r\n]*'?|[^&\s#"'\)\]\}]*)}{$1<redacted>}g;
      $t =~ s{(?i)(authorization[\"']?\s*:\s*)(?=\S)(?!<redacted>)(?![\"']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])[\"']?\s*(?:[,\}\]]|$))(?:[A-Za-z]?\"(?:[^\"\\\r\n]|\\.)*\"?|[A-Za-z]?'(?:[^'\\\r\n]|\\.)*'?|\[[^\]\r\n]*\]?|(?:(?:bearer|basic|digest|negotiate|ntlm|token|apikey)\s+|[A-Za-z][A-Za-z0-9_-]*[ \t]+(?=[^\s,]))?(?![A-Za-z0-9_-]+[\"']?:(?:\s|$))(?:[A-Za-z][A-Za-z0-9_-]*[ \t]*=[ \t]*(?:\"(?:[^\"\\\r\n]|\\.)*\"?|\\\"(?:[^\"\\\r\n]|\\[^\"\r\n])*(?:\\\")?|[^\s,]*)|[^\s,]+)(?:,[ \t]*[A-Za-z][A-Za-z0-9_-]*[ \t]*=[ \t]*(?:\"(?:[^\"\\\r\n]|\\.)*\"?|\\\"(?:[^\"\\\r\n]|\\[^\"\r\n])*(?:\\\")?|[^\s,]*))*)}{$1<redacted>}g;
      $t =~ s{(?i)((?:x-)?(?:api[_-]key|api[_-]token|auth[_-]token)\s*:\s*)(?:(?:bearer|basic|digest|negotiate|ntlm)\s+)?\S+}{$1<redacted>}g;
      $t =~ s{(?i)([\"'](?:(?:[A-Za-z0-9_-]*[_-])?(?:api[_-]?key|token|secret|password|passwd|pwd|access[_-]?token|signature)|[A-Za-z0-9_-]*[A-Za-z0-9][_-](?:key|auth|sig))[\"']\s*:\s*)(?=\S)(?!<redacted>)(?!\s*[\"']?(?:true|false|none|null|nil|yes|no|on|off|enabled|disabled|[01])[\"']?\s*(?:[,\}\]]|$))(?:[A-Za-z]?\"(?:[^\"\\\r\n]|\\.)*\"?|[A-Za-z]?'(?:[^'\\\r\n]|\\.)*'?|\[[^\]\r\n]*\]?|(?:(?:bearer|basic|digest|negotiate|ntlm)[ \t]+)?[^,\}\]\s]*)}{$1<redacted>}g;
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
summary="$(printf '%s' "$summary" | tr '[:cntrl:]' ' ')"
[ -n "$(printf '%s' "$summary" | tr -d '[:space:]')" ] || summary="localm would not start"
if ! summary="$(printf '%s\n' "$summary" | RI_EMAIL="$email" scrub)" || [ -z "$summary" ]; then
  echo "  Cannot remove personal details from the report here (perl is not installed),"
  echo "  so nothing was sent. Please email a short description to ${email}."
  exit 1
fi
os="$(uname -srm 2>/dev/null || echo unknown)"
# Single-line body so JSON escaping stays simple and safe (quotes + backslashes).
esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
body="Filed via the standalone curl reporter (no Python available). OS: ${os}. Details: ${summary}"

echo ""
echo "  ----- this is exactly what will be sent -----"
echo "  title: $summary"
echo "  body:  $body"
echo "  ---------------------------------------------"
# A copy of localm.config.NET_OFF_REMEDY. See test_the_reporters_copy_the_net_off_remedy.
net_off_remedy="Turn it on in Settings > Server & network > Network access, or run: localm config net_mode ask"

# The data dir localm uses: LOCALM_HOME, else the first line of localm-home.cfg,
# else ./home.
ri_home="$(printf '%s' "${LOCALM_HOME:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
if [ -z "$ri_home" ] && [ -f localm-home.cfg ]; then
  ri_home="$(perl -ne 's/\A\xEF\xBB\xBF// if $. == 1; s/^\s+|\s+$//g; if (length) { print; exit }' localm-home.cfg 2>/dev/null)"
fi
[ -n "$ri_home" ] || ri_home="$PWD/home"
case "$ri_home" in "~"|"~/"*) ri_home="${HOME}${ri_home#\~}" ;; esac

# net_mode as localm reads it: LOCALM_NET_MODE, else net_mode in config.json (an
# unset or unknown value means ask); "unknown" when config.json is missing or
# cannot be read, with net_why saying which.
tab="$(printf '\t')"
net_mode="$(printf '%s' "${LOCALM_NET_MODE:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
net_why=""
case "$net_mode" in
  off|ask|allow) ;;
  *)
    net_cfg="$ri_home/config.json"
    if [ ! -f "$net_cfg" ]; then
      net_mode="unknown"; net_why="$net_cfg does not exist"
    else
      net_read="$(perl -MJSON::PP -e '
        my $p = shift;
        open(my $fh, "<:raw", $p) or do { print "unknown\tcould not be read"; exit 0 };
        local $/; my $raw = <$fh>; close $fh;
        $raw =~ s/\A\xEF\xBB\xBF//;
        my $cfg = eval { JSON::PP->new->utf8->decode($raw) };
        if ($@) { print "unknown\tis not valid JSON"; exit 0 }
        if (ref $cfg ne "HASH") { print "unknown\tdoes not hold a settings object"; exit 0 }
        my $m = defined $cfg->{net_mode} ? lc("$cfg->{net_mode}") : "ask";
        $m =~ s/^\s+|\s+$//g;
        print(($m eq "off" || $m eq "ask" || $m eq "allow") ? $m : "ask");
      ' "$net_cfg" 2>/dev/null)" || net_read="unknown${tab}could not be read (perl JSON::PP is missing)"
      net_mode="${net_read%%"$tab"*}"
      [ "$net_mode" = "unknown" ] && net_why="$net_cfg ${net_read#*"$tab"}"
    fi
    ;;
esac

if [ -n "$url" ] && [ "$net_mode" = "off" ]; then
  echo "  Network access is off (net_mode=off), so the report was not sent. ${net_off_remedy}."
  echo "  Email ${email} if you prefer."
  exit 1
fi
if [ -n "$url" ] && [ "$net_mode" = "unknown" ]; then
  echo "  localm's Network access setting could not be read: ${net_why}. Sending connects to the bug-report server."
  printf '  Send it anyway? [y/N]: '
  if ! IFS= read -r ans; then ans=""; echo ""; fi
  ans="$(printf '%s' "$ans" | tr -d '\r[:space:]')"
  case "$ans" in
    [Yy]|[Yy][Ee][Ss]) ;;
    *) echo "  Not sent. Email ${email} if you change your mind."; exit 0 ;;
  esac
else
  printf '  Send this to the maintainer now? [Y/n]: '
  if ! IFS= read -r ans; then
    echo ""
    echo "  Not sent. Email ${email} if you change your mind."
    exit 0
  fi
  case "$ans" in [Nn]*) echo "  Not sent. Email ${email} if you change your mind."; exit 0 ;; esac
fi

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
