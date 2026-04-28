#!/usr/bin/env bash
# Option A: add Entra federated identity for GitHub Actions (one-time per app).
# Prereq: az login, and you have Application Developer or App Admin rights.
#
# Usage:
#   export AZURE_CLIENT_ID=<app registration (client) id from GitHub variable>
#   ./add-github-federated-credential-fdi-lighthouse.sh
#
# Or:
#   ./add-github-federated-credential-fdi-lighthouse.sh c81f9efe-6df2-4608-959e-a427e7a80119
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
APP_ID=${1:-${AZURE_CLIENT_ID:?Set AZURE_CLIENT_ID or pass client id as arg}}
PARAMS="$SCRIPT_DIR/federated-credential-github-fdi-lighthouse-master.json"
echo "Creating federated credential on app $APP_ID from $PARAMS"
az ad app federated-credential create --id "$APP_ID" --parameters @"$PARAMS"
echo "Done. Re-run GitHub Actions workflow 'Run Lighthouse AML pipeline'."
