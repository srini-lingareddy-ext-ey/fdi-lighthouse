# Portal artifact layout (Lighthouse on Azure)

The HTML portal in [../portal/](../portal/) expects a **versioned** folder of JSON + table files + (optional) images, plus a top-level **manifest** that the front end fetches. Generate this folder at the end of a successful training run (or a small post-job step) and **sync** to static hosting (or private blob with SAS/Functions).

## Directory layout (per run)

```text
portal-bundle/
  manifest.json              # required — see schema below
  run.meta.json              # run id, workspace, time
  config/
    config_copy.yml         # copy of the YAML used
  forensics/
    pip_freeze.txt           # or uv.lock + python -m pip list
  forecasts/
    account_forecasts.csv    # from Lighthouse I/O (names may vary; map in manifest)
  drivers/
    driver_rankings_*.json   # optional, if you export from driver analysis
  reconciliation/
    account_reconciliation*.csv
  figures/
    *.png
```

Lighthouse’s on-disk file names are defined under [`lh_v2/io/output/`](../lh_v2/io/output/); the training job (or a wrapper) can **rename** them into the layout above to keep the portal contract stable.

## `manifest.json` (version 1)

| Field | Type | Description |
|--------|------|-------------|
| `version` | string | `"1"` for this document |
| `runId` | string | Azure ML **run** / **job** name or id (audit) |
| `workspace` | string | Optional ML workspace name |
| `createdAtUtc` | string | ISO 8601 timestamp |
| `lighthouse` | object | `gitCommit`, `python` (e.g. `3.14.0`), `lh_v2` package version or path if editable |
| `configHash` | string | Optional short SHA-256 of `config_copy.yml` |
| `modelSummary` | object | Per account: `selectedMethod`, validation metrics if you log them to JSON |
| `files` | object | Keys: logical names; values: **paths relative to the bundle** (e.g. `forecasts/account_forecasts.csv`) |

**Example (minimal):**

```json
{
  "version": "1",
  "runId": "green_grape_fdi1",
  "workspace": "fdi-lighthouse-ml",
  "createdAtUtc": "2026-04-22T12:00:00Z",
  "lighthouse": {
    "python": "3.14.0",
    "gitCommit": "abc1234"
  },
  "configHash": "sha256:...",
  "modelSummary": {
    "volume": { "accountForecastingMethod": "lr_drivers" }
  },
  "files": {
    "accountForecasts": "forecasts/account_forecasts.csv",
    "reconciledForecasts": "reconciliation/reconciled_account_forecasts.csv"
  }
}
```

The [portal `index.html`](../portal/index.html) loads `data/manifest.json` by default (or `?run=<url-to-manifest.json>`) and renders sections when files exist.

## Sync to blob for Static Web App or CDN

- After producing `portal-bundle/`, use `az storage blob upload-batch` to a public `$web` container, or
- [../portal/publish-to-blob.sh](../portal/publish-to-blob.sh) to copy into a **dedicated** storage used only for the portal (recommended over mixing with ML default datastore in production).
