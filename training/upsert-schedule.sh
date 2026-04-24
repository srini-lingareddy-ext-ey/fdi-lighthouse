#!/usr/bin/env bash
# Register (or update) the monthly Lighthouse schedule in Azure ML.
# Usage: ./upsert-schedule.sh <rg> <ws> <acr-fqdn>
set -euo pipefail
RG=${1:-rg-fdi-lighthouse-ml}
WS=${2:-fdi-lighthouse-ml}
ACR=${3:?ACR login server required}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

sed "s|ACR_NAME_PLACEHOLDER.azurecr.io|${ACR}|g" "$SCRIPT_DIR/component.yaml" > "$WORK/component.yaml"
cp "$SCRIPT_DIR/pipeline.yaml" "$WORK/pipeline.yaml"
cp "$SCRIPT_DIR/schedule.yaml" "$WORK/schedule.yaml"
cp "$SCRIPT_DIR/train_entrypoint.py" "$WORK/train_entrypoint.py"

echo "== Creating / updating schedule 'lighthouse-monthly' =="
if az ml schedule show -n lighthouse-monthly -g "$RG" -w "$WS" -o none 2>/dev/null; then
  az ml schedule update -n lighthouse-monthly --file "$WORK/schedule.yaml" -g "$RG" -w "$WS" -o table | tail -5
else
  az ml schedule create --file "$WORK/schedule.yaml" -g "$RG" -w "$WS" -o table | tail -5
fi

echo ""
echo "Enable/disable with: az ml schedule {enable|disable} -n lighthouse-monthly -g $RG -w $WS"
echo "List jobs triggered by this schedule: az ml job list -g $RG -w $WS --query \"[?tags.\\\"azureml.scheduleTrigger.scheduleName\\\"=='lighthouse-monthly']\" -o table"
