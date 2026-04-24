#!/usr/bin/env bash
# Create the smoke job after ACR has the image.
# Usage:  ./run-smoke-job.sh <resource-group> <ml-workspace-name> <acr-login-server>
# Example:  ./run-smoke-job.sh rg-fdi-lighthouse-ml fdi-lighthouse-ml fdiacrx.azurecr.io

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RG=${1:-${RG:-}}
WS=${2:-${WS:-}}
ACR=${3:-${ACR_FQDN:-}}

if [[ -z "$RG" || -z "$WS" || -z "$ACR" ]]; then
  echo "Usage: $0 <resource-group> <ml-workspace-name> <acr-login-server-without-https>"
  echo "  (or set RG, WS, ACR_FQDN in the environment)"
  exit 1
fi

TMP=$(mktemp)
sed "s|ACR_NAME_PLACEHOLDER.azurecr.io|${ACR}|g" "$SCRIPT_DIR/smoke-job.yaml" >"$TMP"
echo "Submitting job (see image line below):"
grep "image:" "$TMP" || true
az ml job create -f "$TMP" -g "$RG" -w "$WS"
rm -f "$TMP"
