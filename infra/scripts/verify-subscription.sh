#!/usr/bin/env bash
# Read-only: verify Azure CLI and inventory subscription before deploying Bicep.
# Usage: ./verify-subscription.sh [resource-group-name]

set -euo pipefail

az account show --output table
echo ""
echo "=== Resource groups (first 30) ==="
az group list -o table 2>&1 | head -35
echo ""
if ! az extension show -n ml -o table 2>/dev/null; then
  echo "Tip: install ML v2 extension:  az extension add -n ml -y"
else
  echo "=== Azure ML extension ==="
  az extension show -n ml -o table
fi
echo ""

if [[ -n "${1:-}" ]]; then
  echo "=== Resources in RG: $1 ==="
  az resource list -g "$1" -o table 2>/dev/null || echo "(RG not found or no access)"
  echo ""
  if az extension show -n ml &>/dev/null; then
    echo "=== Azure ML workspaces (subscription) ==="
    az ml workspace list 2>/dev/null || true
  fi
fi

echo "Done. Use this output to pick subscription, region, and whether to create a new RG for Lighthouse."
