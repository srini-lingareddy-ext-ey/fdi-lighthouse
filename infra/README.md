# Azure infrastructure (Lighthouse + Azure ML)

This directory deploys a **resource group**–scoped stack: **Log Analytics + Application Insights**, **Storage account**, **Key Vault**, **Azure Container Registry (ACR)**, **Azure Machine Learning workspace** (linked to the above), **AmlCompute** (scale-to-zero), and **RBAC** for the workspace managed identity (Key Vault Secrets User, Storage Blob Data Contributor, AcrPull).

- **Bicep templates:** [bicep/main.bicep](bicep/main.bicep) and [bicep/modules/](bicep/modules/)
- **Verify subscription (read-only):** [scripts/verify-subscription.sh](scripts/verify-subscription.sh)

## Prerequisites

- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) (`az`) with a login: `az login`
- Bicep: `az bicep version` (or `az bicep install`)
- Permission: **Contributor** on the target resource group (or subscription-level for provider registration)

## 1) Verify subscription

From the **repository root**:

```bash
chmod +x infra/scripts/verify-subscription.sh
./infra/scripts/verify-subscription.sh
./infra/scripts/verify-subscription.sh rg-fdi-lighthouse-ml
```

`az extension add -n ml -y` is optional, for `az ml workspace list`.

## 2) Create a resource group

```bash
az group create -n rg-fdi-lighthouse-ml -l eastus
```

## 3) Deploy Bicep (idempotent)

```bash
cd bicep
az deployment group create \
  -g rg-fdi-lighthouse-ml \
  -f main.bicep \
  -p @main.parameters.json \
  --name lighthouse-ml-$(date +%Y%m%d%H%M)
```

Read outputs:

```bash
az deployment group show -g rg-fdi-lighthouse-ml -n <deploymentName> --query properties.outputs -o json
```

Useful outputs: `acrLoginServer`, `mlWorkspaceId` / workspace name, `storageAccountName`, `computeName`.

## 4) Next steps

- **Docker (Python 3.14) + ACR build:** [docker/README.md](docker/README.md)
- **CSV + data assets + `az ml job`:** [docs/azure-ml-data-and-jobs.md](docs/azure-ml-data-and-jobs.md)
- **HTML portal bundle (manifest, Static Web App):** [../portal/README.md](../portal/README.md)

## What is not in Bicep

- Private endpoints / VNet
- Azure Static Web App (portal front end)
- Budgets, action groups, or diagnostic setting exports (add for production)

## Troubleshooting

- **Providers:** `az provider register -n Microsoft.MachineLearningServices --wait` (and `Microsoft.Insights`, `Microsoft.OperationalInsights`, `Microsoft.KeyVault`, `Microsoft.Storage`, `Microsoft.ContainerRegistry` as needed)
- **Redeploy:** Bicep update is idempotent. Global names (Key Vault) remain reserved while soft-deleted; avoid deleting the RG frequently in dev
- **Role assignment:** If deployment fails on `roleAssignments` duplicate, the subscription may have an existing assignment with the same `name` GUID; delete the conflicting role assignment in the portal or use a new RG
