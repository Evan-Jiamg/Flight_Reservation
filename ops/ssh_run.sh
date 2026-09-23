#!/bin/bash
# Usage: ssh_run.sh <host> <remote_script_file> [upload_tgz]
# Sends a remote bash script (and optionally a tgz payload exposed as $PAYLOAD_B64)
# over ONE ssh connection. Always waits 15s before connecting (server pacing rule).
set -u
HOST=$1; SCRIPT=$2; TGZ=${3:-}
sleep 15
S=$(base64 -w0 "$SCRIPT")
if [ -n "$TGZ" ]; then P=$(base64 -w0 "$TGZ"); else P=""; fi
timeout ${SSH_TIMEOUT:-110} ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" "export PAYLOAD_B64='$P'; F=/tmp2/mzjiang_usersim/.cc_remote_\$\$.sh; echo $S | base64 -d > \$F; bash \$F; rc=\$?; rm -f \$F; exit \$rc"
