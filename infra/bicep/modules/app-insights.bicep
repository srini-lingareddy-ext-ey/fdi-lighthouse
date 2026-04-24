@description('Azure region; must support Log Analytics + App Insights.')
param location string

@description('Application Insights + Log Analytics name prefix (alphanumeric, unique in subscription for LA).')
param name string

@description('Resource tags (cost center, env, etc.).')
param tags object = {}

// Log Analytics workspace (required for workspace-based App Insights in many regions)
resource logWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${name}-laws'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: name
  location: location
  kind: 'web'
  tags: tags
  properties: {
    Application_Type: 'web'
    IngestionMode: 'LogAnalytics'
    WorkspaceResourceId: logWorkspace.id
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

@description('Application Insights component resource id for the ML workspace.')
output applicationInsightsId string = appInsights.id

@description('Log Analytics workspace resource id (optional for diagnostics).')
output logAnalyticsId string = logWorkspace.id
