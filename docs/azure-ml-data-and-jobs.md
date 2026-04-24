# Azure ML: demo CSVs, data assets, and job inputs (v2)

After [Bicep deploy](../infra/README.md) and an [ACR image](../docker/README.md), place **account** and **driver** CSVs on the **default blob datastore** (or a custom one), register **Data assets (uri_file / uri_folder)**, and mount them as **job inputs** so the training `command` sees paths like `$INPUT_ACCOUNTS` / Azure ML v2 `inputs` mapping.

## 1) Default datastore (built-in)

The Machine Learning workspace default datastore is the **Storage account** created with the workspace (Bicep output: `storageAccountName`). The default container is usually `azureml-blobstore-*` and the default datastore name is `workspaceblobstore` or `workspaceartifactstore` depending on the API version; confirm in **Studio** → **Data** → **Data stores**.

Get it via CLI:

```bash
az extension add -n ml -y
az ml datastore show -n workspaceblobstore -g "$RG" -w "$WS" 2>/dev/null || \
az ml datastore list -g "$RG" -w "$WS" -o table
```

## 2) Upload `demo/accounts.csv` and `demo/drivers.csv`

Replace storage account, container, and path to match your layout. Example using **Azure Storage** and a container `lighthouse-raw` (create the container first in the portal or with CLI):

```bash
export RG="rg-fdi-lighthouse-ml"
# Storage account from Bicep / deployment output (not the long ml workspace default name — use the *resource* name)
export STG="<storageAccountName-from-deployment>"

# Create a container (once)
az storage container create -n lighthouse-raw --account-name "$STG" --auth-mode login

# Upload from repo
REPO_ROOT="$(git rev-parse --show-toplevel)"
az storage blob upload -f "$REPO_ROOT/demo/accounts.csv" \
  -c lighthouse-raw -n demo/accounts.csv --account-name "$STG" --auth-mode login
az storage blob upload -f "$REPO_ROOT/demo/drivers.csv" \
  -c lighthouse-raw -n demo/drivers.csv --account-name "$STG" --auth-mode login
```

If the storage account has **public network** rules, use `az storage account update` to allow your IP, or use **Entra** auth as above with `--auth-mode login` (requires *Storage Blob Data Contributor* for your user, or a SAS for scripts).

A helper script: [../infra/scripts/upload-demo-csvs.sh](../infra/scripts/upload-demo-csvs.sh) (takes `RG` and your storage name).

## 3) Register a Data asset (v2) pointing at the folder or files

**Folder** (URI folder) on `azureml://` datastore:

```bash
az ml data create \
  -g "$RG" -w "$WS" -n demo-lh-csv \
  --type uri_folder \
  --path "azureml://datastores/$DATASTORE/paths/lighthouse-raw"
```

Get `DATASTORE` and `path` to match the blob path you used. Simpler path for **file**-level assets:

```bash
az ml data create \
  -g "$RG" -w "$WS" -n demo-accounts-csv \
  --type uri_file \
  --path "azureml://datastores/${DATASTORE}/paths/lighthouse-raw/demo/accounts.csv"
```

Replace `DATASTORE` (often `workspaceblobstore`) after checking `az ml datastore list`.

## 4) Wire a training job: mount `inputs` and call Lighthouse

A future **entry script** (not part of the core `lh_v2` library) will:

- Read `sys.argv` or **environment** variables (or `AZURE_ML_` paths if you copy assets to local mount).
- In Azure ML v2, **input** is mounted; use `path` in `inputs` in the **job yaml** to bind `accounts.csv` to `/mnt/accounts/accounts.csv` and pass that to `load_data_driver_ranking`. See: [Command job with data inputs](https://learn.microsoft.com/azure/machine-learning/how-to-read-write-data-v2#consume-data-assets-in-jobs) (use current ML docs for the exact `inputs` / `data` field names for your `az ml` / SDK version).

**Sketch (YAML, illustrative):**

```yaml
# Illustrative only — follow the commandJob.schema.json for your CLI version
type: command
code: <path to your small train script>   # e.g. repo with azureml_entry/train.py
command: |
  python azureml_entry/train.py --accounts "${{inputs.accounts}}" --drivers "${{inputs.drivers}}" --config "${{inputs.config}}"
compute: azureml:lh-cpu
environment:
  image: <acr>.azurecr.io/fdi-lighthouse:py314
inputs:
  accounts:
    type: uri_file
    path: azureml:demo-accounts-csv:1
  drivers:
    type: uri_file
    path: azureml:demo-drivers-csv:1
  config:
    type: uri_file
    path: <path to config yml in blob or code snapshot>
```

**Azure ML 2.0** sometimes references assets by `azureml:asset_name:version` — re-check with `az ml data -h` for the exact `path` for your `az ml` version. If your CLI only supports `path` in `create`, the Studio UI is another option for wiring inputs.

## 5) Write outputs and publish to the HTML portal

- Enable Lighthouse **saving of artifacts** in your job (`b_save_info`, output dir), then copy or sync the **output folder** to a `portal-assets` blob prefix and generate a **`manifest.json`**; see [artifact-layout.md](artifact-layout.md) and the [../portal/](../portal/) app.

## References

- [Data in Azure Machine Learning](https://learn.microsoft.com/azure/machine-learning/concept-data)
- [CLI v2 `az ml` reference](https://learn.microsoft.com/cli/azure/ml)
