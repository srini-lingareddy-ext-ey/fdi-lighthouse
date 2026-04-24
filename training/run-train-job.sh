#!/usr/bin/env bash
# Submit the Lighthouse training job to Azure ML.
# Usage:
#   ./run-train-job.sh <resource-group> <workspace> <acr-fqdn> [--wait]
# Example:
#   ./run-train-job.sh rg-fdi-lighthouse-ml fdi-lighthouse-ml acregfdilhreg6kqxjx2266ii6.azurecr.io --wait

set -euo pipefail
RG=${1:-rg-fdi-lighthouse-ml}
WS=${2:-fdi-lighthouse-ml}
ACR=${3:?ACR login server required (e.g. myacr.azurecr.io)}
WAIT_FLAG=${4:-}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TMP_JOB=$(mktemp)
trap 'rm -f "$TMP_JOB"' EXIT

sed "s|ACR_NAME_PLACEHOLDER.azurecr.io|${ACR}|g" "$SCRIPT_DIR/train-job.yaml" > "$TMP_JOB"

echo "Submitting training job (code=${SCRIPT_DIR}, image=${ACR}/fdi-lighthouse:py314)..."
JOB_NAME=$(az ml job create --file "$TMP_JOB" -g "$RG" -w "$WS" --set code="$SCRIPT_DIR" --query name -o tsv)
STUDIO_URL=$(az ml job show -n "$JOB_NAME" -g "$RG" -w "$WS" --query "services.Studio.endpoint" -o tsv 2>/dev/null || echo "")
echo "Submitted:  $JOB_NAME"
echo "Studio URL: $STUDIO_URL"

if [[ "$WAIT_FLAG" == "--wait" ]]; then
  echo "Waiting for job to complete..."
  az ml job stream -n "$JOB_NAME" -g "$RG" -w "$WS" || true
  STATUS=$(az ml job show -n "$JOB_NAME" -g "$RG" -w "$WS" --query status -o tsv)
  echo "Final status: $STATUS"
  [[ "$STATUS" == "Completed" ]] || exit 2
fi

echo "$JOB_NAME"
