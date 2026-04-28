# Option A: add Entra federated identity for GitHub Actions (one-time per app).
# Prereq: az login, Application Developer or App Admin rights.
#
# Usage (from repo root, PowerShell):
#   cd infra\scripts
#   .\add-github-federated-credential-fdi-lighthouse.ps1
#   .\add-github-federated-credential-fdi-lighthouse.ps1 c81f9efe-6df2-4608-959e-a427e7a80119

param(
    [string] $AppId = $env:AZURE_CLIENT_ID
)

$ErrorActionPreference = 'Stop'
if (-not $AppId) {
    Write-Error "Set AZURE_CLIENT_ID or pass client id: .\add-github-federated-credential-fdi-lighthouse.ps1 <client-id>"
}
$json = Join-Path $PSScriptRoot 'federated-credential-github-fdi-lighthouse-master.json'
Write-Host "Creating federated credential on app $AppId using $json"
az ad app federated-credential create --id $AppId --parameters "@$json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done. Re-run GitHub Actions workflow 'Run Lighthouse AML pipeline'."
