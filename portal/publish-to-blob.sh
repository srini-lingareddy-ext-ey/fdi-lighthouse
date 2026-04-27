#!/usr/bin/env bash
# Enable static website hosting on the storage account, copy training artifacts
# into the $web/data/ folder, and publish the static HTML portal.
#
# Usage (from repo root):
#   STG=<storageAccountName> ARTIFACTS=<path-to-run-outputs> ./portal/publish-to-blob.sh
#
# Example:
#   STG=sfdilhst6kqxjx2266ii6 \
#     ARTIFACTS=/tmp/lh-artifacts/named-outputs/artifacts \
#     ./portal/publish-to-blob.sh
#
# The static site will be served at the storage account's primary web endpoint,
# printed at the end of this script.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
STG=${STG:?Set STG to the storage account name}
ARTIFACTS=${ARTIFACTS:-}

echo "== Enabling static website on $STG =="
az storage blob service-properties update \
  --account-name "$STG" \
  --static-website \
  --index-document index.html \
  --404-document index.html \
  --auth-mode login \
  --only-show-errors 1>/dev/null

echo "== Uploading HTML shell to \$web =="
az storage blob upload \
  --account-name "$STG" --auth-mode login --only-show-errors \
  --container-name '$web' \
  --file "$SCRIPT_DIR/index.html" \
  --name index.html --overwrite 1>/dev/null

if [[ -n "$ARTIFACTS" && -d "$ARTIFACTS" ]]; then
  if [[ ! -f "$ARTIFACTS/manifest.json" ]]; then
    echo "warn: $ARTIFACTS/manifest.json not found; uploading anyway."
  fi
  echo "== Uploading run artifacts to \$web/data/ =="
  az storage blob upload-batch \
    --account-name "$STG" --auth-mode login --only-show-errors \
    --destination '$web' --destination-path data \
    --source "$ARTIFACTS" --overwrite 1>/dev/null
else
  echo "(No ARTIFACTS path provided — skipping data upload. Portal will show 'manifest not found' until data/ is populated.)"
fi

WEB_URL=$(az storage account show -n "$STG" --query "primaryEndpoints.web" -o tsv)
echo ""
echo "Portal published."
echo "  URL         : ${WEB_URL}"
echo "  Manifest    : ${WEB_URL}data/manifest.json"
echo ""
echo "Open:   ${WEB_URL}"
