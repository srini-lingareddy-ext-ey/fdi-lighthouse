#!/usr/bin/env bash
# Submit the Lighthouse pipeline job (one-off) to Azure ML.
# Usage: ./run-pipeline.sh <rg> <ws> <acr-fqdn>
set -euo pipefail
RG=${1:-rg-fdi-lighthouse-ml}
WS=${2:-fdi-lighthouse-ml}
ACR=${3:?ACR login server required}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TMP_COMP=$(mktemp); TMP_PIPE=$(mktemp)
trap 'rm -f "$TMP_COMP" "$TMP_PIPE"' EXIT

sed "s|ACR_NAME_PLACEHOLDER.azurecr.io|${ACR}|g" "$SCRIPT_DIR/component.yaml" > "$TMP_COMP"
# pipeline.yaml references ./component.yaml relatively; copy the patched component next to it
cp "$SCRIPT_DIR/pipeline.yaml" "$TMP_PIPE"
PATCHED_DIR=$(mktemp -d)
cp "$TMP_COMP" "$PATCHED_DIR/component.yaml"
cp "$SCRIPT_DIR/pipeline.yaml" "$PATCHED_DIR/pipeline.yaml"
cp "$SCRIPT_DIR/train_entrypoint.py" "$PATCHED_DIR/"

JOB_NAME=$(az ml job create --file "$PATCHED_DIR/pipeline.yaml" -g "$RG" -w "$WS" --query name -o tsv)
echo "Pipeline submitted: $JOB_NAME"
