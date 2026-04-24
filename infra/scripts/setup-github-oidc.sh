#!/usr/bin/env bash
# One-time bootstrap: create an Azure AD app registration + service principal and
# wire a GitHub OIDC federated credential so GitHub Actions can authenticate to
# Azure without storing any long-lived secrets. Grants the SP the roles needed
# to push to ACR and submit AML jobs in the Lighthouse resource group.
#
# Usage:
#   ./setup-github-oidc.sh <github-org/repo> [branch] [rg] [app-name]
#
# Example:
#   ./setup-github-oidc.sh ey-fdi/fdi-lighthouse master rg-fdi-lighthouse-ml fdi-lighthouse-gha
#
# Outputs the three values to configure as GitHub repository variables:
#   AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID
set -euo pipefail

REPO=${1:?"github org/repo required, e.g. ey-fdi/fdi-lighthouse"}
BRANCH=${2:-master}
RG=${3:-rg-fdi-lighthouse-ml}
APP_NAME=${4:-fdi-lighthouse-gha}

SUB_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)

echo "== Creating / reusing AAD app '$APP_NAME' =="
APP_ID=$(az ad app list --display-name "$APP_NAME" --query "[0].appId" -o tsv || true)
if [[ -z "${APP_ID:-}" ]]; then
  APP_ID=$(az ad app create --display-name "$APP_NAME" --query appId -o tsv)
fi
echo "appId=$APP_ID"

echo "== Ensuring service principal for app =="
SP_OID=$(az ad sp list --filter "appId eq '$APP_ID'" --query "[0].id" -o tsv || true)
if [[ -z "${SP_OID:-}" ]]; then
  SP_OID=$(az ad sp create --id "$APP_ID" --query id -o tsv)
fi
echo "sp.objectId=$SP_OID"

echo "== Granting roles on resource group '$RG' =="
RG_SCOPE=$(az group show -n "$RG" --query id -o tsv)
for ROLE in "Contributor" "AcrPush" "AzureML Data Scientist"; do
  az role assignment create \
    --assignee-object-id "$SP_OID" --assignee-principal-type ServicePrincipal \
    --role "$ROLE" --scope "$RG_SCOPE" -o none 2>/dev/null || true
done

echo "== Registering federated credentials =="
add_fic() {
  local name=$1 subject=$2
  az ad app federated-credential create --id "$APP_ID" --parameters "{
    \"name\": \"$name\",
    \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"$subject\",
    \"audiences\": [\"api://AzureADTokenExchange\"]
  }" -o none 2>/dev/null || echo "  (already exists: $name)"
}
add_fic "gh-${BRANCH}"       "repo:${REPO}:ref:refs/heads/${BRANCH}"
add_fic "gh-pull-request"    "repo:${REPO}:pull_request"
add_fic "gh-workflow-dispatch" "repo:${REPO}:ref:refs/heads/${BRANCH}"

cat <<EOF

Done. Configure these as GitHub repository *Variables* (Settings -> Secrets and variables -> Actions -> Variables):
  AZURE_CLIENT_ID=$APP_ID
  AZURE_TENANT_ID=$TENANT_ID
  AZURE_SUBSCRIPTION_ID=$SUB_ID

Workflows use id-token: write + azure/login@v2 with those variables.
EOF
