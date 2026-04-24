param location string
param name string
param tags object = {}

@description('Use RBAC for Key Vault; ML workspace managed identity is granted via role assignments in main.bicep.')
param enableRbacAuthorization bool = true

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enabledForTemplateDeployment: true
    enableRbacAuthorization: enableRbacAuthorization
    enableSoftDelete: true
    // Tenant policy may require purge protection; keep enabled by default
    enablePurgeProtection: true
  }
}

output id string = keyVault.id
output name string = keyVault.name
