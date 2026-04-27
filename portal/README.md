# HTML portal (forecasts, audit, file links)

Static **read-only** UI that loads a **`manifest.json`** compatible with [docs/artifact-layout.md](../docs/artifact-layout.md). Training or a post-run script should place CSVs, plots, and a manifest under a folder (e.g. `portal-bundle/`) and publish to hosting.

| Section | Data source |
|---------|-------------|
| **Audit** | `manifest.json` — `runId`, `createdAtUtc`, `workspace`, `lighthouse` (python, git), `configHash` |
| **Forecasts / rankings / reconciliation** | Linked via `manifest.files` to CSV/JSON/PNG under the same base URL |
| **Charts** | `figures/*.png` (or your paths in `files`) if you add chart exports from the job |

## Local preview

```bash
cd portal
python -m http.server 8080
# Open http://127.0.0.1:8080/ — loads data/manifest.json
```

`file://` may block `fetch()`; use a local server as above.

**Custom manifest URL:** `index.html?manifest=https://example.blob.../run123/manifest.json` (blob must have **CORS** allowed for your portal origin, or use a same-origin API).

## Azure Static Web Apps (typical)

1. Commit `portal/` (or the built `dist/` after adding real artifacts in CI).
2. Create a [Static Web App](https://learn.microsoft.com/azure/static-web-apps/getting-started) in the Azure portal, connect a Git branch, and set the **app location** to `portal` (or `dist`).
3. Optional: [Azure Static Web App APIs](https://learn.microsoft.com/azure/static-web-apps/apis-overview) in `api/` to generate SAS or proxy private blob (not included here).

## Azure Storage static website (blob `$web`)

1. Create or reuse a **Storage account** (dedicated to the public portal, not the raw ML data).
2. Enable [static website hosting](https://learn.microsoft.com/azure/storage/blobs/static-website-host) on the account; upload `index.html`, `data/`, and asset folders.
3. [publish-to-blob.sh](publish-to-blob.sh) is a **batch upload** example (adjust `CONTAINER` to `$web` in the Azure CLI, escaping may be required in your shell).

## Copy from a completed ML run

1. In **Azure Machine Learning** studio, open the **Job** → **Outputs + logs**; download the folder your entry script wrote.
2. Merge with `index.html` + `data/manifest.json` into one directory.
3. Edit `manifest.json` to set `files` to relative paths for the downloaded CSV/PNG.
4. Publish to SWA or blob as above.

For automation, a pipeline step can **sync** the job’s `azureml-logs` or custom output path to the portal storage (see the plan: optional copy job).

## Security

- Do not publish **production secrets** in the manifest. Keep `config_copy.yml` non-secret or redact.
- If data is not public, use a **Function** with managed identity, **SAS** with short TTL, or **Entra**-protected Static Web App (if available in your plan).
