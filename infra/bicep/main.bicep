// Deploy at resource group scope. Create the RG first:
//   az group create -n rg-fdi-lighthouse-ml -l eastus2
//   az deployment group create -g rg-fdi-lighthouse-ml -f main.bicep -p @main.parameters.json

targetScope = 'resourceGroup'

@minLength(2)
@maxLength(10)
@description('Short prefix for global names (storage, key vault, ACR).')
param namePrefix string

@description('Region for all resources; pick one that supports Azure ML and ACR.')
param location string = resourceGroup().location

@description('Name of the Azure Machine Learning workspace resource (unique per resource group; valid ARM name).')
param mlWorkspaceName string

@description('AmlCompute cluster name; keep short (e.g. lh-cpu).')
@minLength(2)
@maxLength(16)
param computeClusterName string = 'lh-cpu'

@description('VM size for training; CPU is fine for Lighthouse unless you add GPU models.')
param vmSize string = 'Standard_DS3_v2'

@minValue(0)
@maxValue(50)
@description('Max number of cluster nodes; min 0 to scale to zero when idle.')
param maxNodeCount int = 4

@description('Key Vault Secrets User (built-in role definition GUID; override if your tenant redefines roles).')
param keyVaultRoleDefinitionId string = '4633458b-17de-408a-b874-0445c86b69e6'

@description('Storage Blob Data Contributor (built-in role definition GUID).')
param storageBlobRoleDefinitionId string = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'

@description('AcrPull (built-in role definition GUID).')
param acrPullRoleDefinitionId string = '7f951dda-4ed3-4680-a7ca-43fe172d538d'

param tags object = {}

@description('ACR: enable admin for quick `az acr login` (disable in lock-down environments).')
param acrAdminUserEnabled bool = true

var nameHash = uniqueString(subscription().id, resourceGroup().id, namePrefix, mlWorkspaceName)
// Storage: 3–24, lowercase, must start with a letter
var storageAccountName = toLower(
  take('s${replace('${namePrefix}st${nameHash}', '-', '')}', 24)
)
// Key vault: 3–24, must start with a letter
var keyVaultName = toLower(
  take('k${replace('${namePrefix}v${nameHash}', '-', '')}', 24)
)
// ACR: 5–50, letter start; pad so take() can never be shorter than 5
var acrName = toLower(
  take('acreg${replace('${namePrefix}reg${nameHash}', '-', '')}', 50)
)
var appInsightsName = toLower('${namePrefix}ai${take(nameHash, 7)}')

module appInsights 'modules/app-insights.bicep' = {
  name: 'app-insights-deploy'
  params: {
    location: location
    name: appInsightsName
    tags: tags
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storage-deploy'
  params: {
    location: location
    name: storageAccountName
    tags: tags
  }
}

module keyVault 'modules/key-vault.bicep' = {
  name: 'keyvault-deploy'
  params: {
    location: location
    name: keyVaultName
    tags: tags
  }
}

module acr 'modules/acr.bicep' = {
  name: 'acr-deploy'
  params: {
    location: location
    name: acrName
    tags: tags
    acrSku: 'Basic'
    adminUserEnabled: acrAdminUserEnabled
  }
}

resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: mlWorkspaceName
  location: location
  kind: 'Default'
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'Lighthouse (fdi-lighthouse) batch training'
    description: 'Lighthouse lh_v2 CSV batch jobs; custom Python 3.14 from ACR.'
    keyVault: keyVault.outputs.id
    applicationInsights: appInsights.outputs.applicationInsightsId
    storageAccount: storage.outputs.id
    containerRegistry: acr.outputs.id
    publicNetworkAccess: 'Enabled'
  }
}

resource amlCompute 'Microsoft.MachineLearningServices/workspaces/computes@2024-04-01' = {
  parent: mlWorkspace
  identity: {
    type: 'SystemAssigned'
  }
  name: computeClusterName
  location: location
  tags: tags
  properties: {
    computeType: 'AmlCompute'
    description: 'CPU for monthly Lighthouse training'
    properties: {
      vmSize: vmSize
      osType: 'Linux'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: maxNodeCount
        nodeIdleTimeBeforeScaleDown: 'PT300S'
      }
      enableNodePublicIp: true
    }
  }
}

resource keyVaultR 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource storageR 'Microsoft.Storage/storageAccounts@2023-01-01' existing = {
  name: storageAccountName
}

resource acrR 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' existing = {
  name: acrName
}

resource raKeyVault 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVaultR.id, mlWorkspace.id, 'KeyVaultSecretsUser')
  scope: keyVaultR
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      keyVaultRoleDefinitionId
    )
    principalId: mlWorkspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource raStorage 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageR.id, mlWorkspace.id, 'BlobDataContributor')
  scope: storageR
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      storageBlobRoleDefinitionId
    )
    principalId: mlWorkspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource raAcr 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acrR.id, mlWorkspace.id, 'AcrPull')
  scope: acrR
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleDefinitionId)
    principalId: mlWorkspace.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Compute cluster also needs access to storage so data-capability can mount uri_file/uri_folder
// inputs using its own MI (prevents NoIdentityOnCompute during job startup).
resource raStorageCompute 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageR.id, amlCompute.id, 'BlobDataContributor')
  scope: storageR
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      storageBlobRoleDefinitionId
    )
    principalId: amlCompute.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource raAcrCompute 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acrR.id, amlCompute.id, 'AcrPull')
  scope: acrR
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleDefinitionId)
    principalId: amlCompute.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

@description('Name of the ML workspace (e.g. `az ml -w` / `default_azure_credential` resource name).')
output mlWorkspaceName string = mlWorkspaceName

@description('Azure Machine Learning resource id (for `az ml workspace` / SDK).')
output mlWorkspaceId string = mlWorkspace.id

@description('AmlCompute resource name in the workspace (see output computeName).')
output computeName string = computeClusterName

@description('ACR login server (e.g. myreg.azurecr.io) for `docker tag` and `az acr build`')
output acrLoginServer string = acr.outputs.loginServer

@description('ACR resource name')
output acrNameOut string = acr.outputs.name

@description('Default backing storage account name (workspace default datastore is under it).')
output storageAccountName string = storageAccountName

@description('Workspace system-assigned managed identity principalId (troubleshoot RBAC).')
output mlWorkspaceIdentityPrincipalId string = mlWorkspace.identity.principalId

@description('Application Insights resource id (monitoring in portal).')
output appInsightsId string = appInsights.outputs.applicationInsightsId

@description('Resource group of this deployment.')
output resourceGroupName string = resourceGroup().name
