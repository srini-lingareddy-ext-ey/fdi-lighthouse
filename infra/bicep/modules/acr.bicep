param location string
param name string
param tags object = {}

@description('Use Basic for dev; Standard/Premium for geo-rep or webhooks.')
param acrSku string = 'Basic'

@description('Allow az acr login for manual pushes; set false in production and use ACR build + AcrPull on MI only.')
param adminUserEnabled bool = true

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: name
  location: location
  tags: tags
  sku: { name: acrSku }
  properties: {
    adminUserEnabled: adminUserEnabled
    publicNetworkAccess: 'Enabled'
    // Required for ACR on Azure ML: premium features like zone redundancy can be set when sku allows
  }
}

output id string = acr.id
output name string = acr.name
@description('ACR login server, e.g. myregistry.azurecr.io')
output loginServer string = acr.properties.loginServer
