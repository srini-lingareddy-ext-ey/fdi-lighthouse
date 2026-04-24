#!/usr/bin/env bash
# Register a workspace datastore and versioned v2 data assets for the demo CSVs/config.
# Idempotent: re-running updates the assets.
#
# Usage:  RG=rg-fdi-lighthouse-ml WS=fdi-lighthouse-ml STG=sfdilhst6kqxjx2266ii6 \
#           ./register-data-assets.sh

set -euo pipefail
RG=${RG:-rg-fdi-lighthouse-ml}
WS=${WS:-fdi-lighthouse-ml}
STG=${STG:?set STG to the storage account name}
CONTAINER=${CONTAINER:-lighthouse-raw}
DATASTORE=${DATASTORE:-lighthouse_raw}
VERSION=${VERSION:-1}

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

cat > "$WORK_DIR/datastore.yml" <<EOF
\$schema: https://azuremlschemas.azureedge.net/latest/azureBlob.schema.json
name: ${DATASTORE}
type: azure_blob
description: Raw CSV + config files for Lighthouse (demo)
account_name: ${STG}
container_name: ${CONTAINER}
EOF

echo "== Registering datastore ${DATASTORE} =="
az ml datastore create --file "$WORK_DIR/datastore.yml" -g "$RG" -w "$WS" 2>&1 | tail -10 || \
  az ml datastore update --file "$WORK_DIR/datastore.yml" -g "$RG" -w "$WS" 2>&1 | tail -5

register_asset () {
  local name=$1 kind=$2 relpath=$3 desc=$4
  cat > "$WORK_DIR/$name.yml" <<EOF
\$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: ${name}
version: "${VERSION}"
type: ${kind}
description: ${desc}
path: azureml://datastores/${DATASTORE}/paths/${relpath}
EOF
  echo "== Data asset ${name}:${VERSION} -> ${kind} ${relpath} =="
  az ml data create --file "$WORK_DIR/$name.yml" -g "$RG" -w "$WS" 2>&1 | tail -3 || true
}

register_asset "lh-accounts" "uri_file"   "demo/accounts.csv" "Demo accounts fact CSV"
register_asset "lh-drivers"  "uri_file"   "demo/drivers.csv"  "Demo drivers CSV"
register_asset "lh-config"   "uri_file"   "demo/config.yml"   "Demo lh_v2 config"
register_asset "lh-demo"     "uri_folder" "demo"              "Demo folder with accounts/drivers/config"

echo ""
echo "Registered data assets (lh-accounts, lh-drivers, lh-config, lh-demo) @ v${VERSION}"
