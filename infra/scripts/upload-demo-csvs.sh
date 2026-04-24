#!/usr/bin/env bash
# Upload demo/accounts.csv and demo/drivers.csv to a user-defined container in the
# Bicep-created storage account (or any storage account in the same subscription).
#
# Prereq:  az login  and  Storage Blob Data access (e.g. Contributor on the RG) or a SAS
#
# Usage:  STG=storagename RG=rg-fdi-lighthouse-ml  ./upload-demo-csvs.sh
#   Optional:  CONTAINER=lighthouse-raw  PREFIX=demo
#
# After upload, use Azure ML Studio or `az ml data create` to register a uri_folder/uri_file
# pointing at `azureml://...` — see docs/azure-ml-data-and-jobs.md

set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STG=${STG:-}
RG=${RG:-}
CONTAINER=${CONTAINER:-lighthouse-raw}
PREFIX=${PREFIX:-demo}

if [[ -z "$STG" ]]; then
  if [[ -n "$RG" ]]; then
    STG=$(az storage account list -g "$RG" --query "[0].name" -o tsv 2>/dev/null) || true
  fi
fi
if [[ -z "$STG" ]]; then
  echo "Set STG to the storage account name (Bicep output storageAccountName) or set RG to pick the first in the RG."
  exit 1
fi

az storage container create -n "$CONTAINER" --account-name "$STG" --auth-mode login --only-show-errors
az storage blob upload -f "$REPO_ROOT/demo/accounts.csv" -c "$CONTAINER" -n "$PREFIX/accounts.csv" --account-name "$STG" --auth-mode login --only-show-errors --overwrite
az storage blob upload -f "$REPO_ROOT/demo/drivers.csv" -c "$CONTAINER" -n "$PREFIX/drivers.csv" --account-name "$STG" --auth-mode login --only-show-errors --overwrite
az storage blob upload -f "$REPO_ROOT/demo/config.yml"   -c "$CONTAINER" -n "$PREFIX/config.yml"   --account-name "$STG" --auth-mode login --only-show-errors --overwrite
echo "Uploaded: $CONTAINER/$PREFIX/{accounts.csv, drivers.csv, config.yml}  on account  $STG"
echo "Next: register a Data asset (az ml data create) — see docs/azure-ml-data-and-jobs.md"
