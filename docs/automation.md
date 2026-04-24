# Automation & Scheduling

This doc describes the two automation layers on top of the Azure ML +
Lighthouse (`lh_v2`) platform.

## 1. Monthly forecast — Azure ML pipeline + schedule

Artifacts under `training/`:

| File                         | Purpose                                                              |
| ---------------------------- | -------------------------------------------------------------------- |
| `component.yaml`             | Reusable AML command component wrapping `train_entrypoint.py`.       |
| `pipeline.yaml`              | Pipeline job wiring the `lh-accounts@latest` / `lh-drivers@latest` / `lh-config@latest` data assets to the component. |
| `schedule.yaml`              | `RecurrenceTrigger` (monthly, 1st @ 02:00 UTC) that submits `pipeline.yaml`. |
| `upsert-schedule.sh`         | Idempotent helper — patches the ACR FQDN into `component.yaml`, then creates or updates the schedule in the workspace. |
| `run-pipeline.sh`            | One-off pipeline submission (same patching flow) for ad‑hoc runs.    |

### Re‑register / update the schedule

```bash
./training/upsert-schedule.sh \
  rg-fdi-lighthouse-ml \
  fdi-lighthouse-ml \
  acregfdilhreg6kqxjx2266ii6.azurecr.io
```

Useful commands once registered:

```bash
az ml schedule list    -g rg-fdi-lighthouse-ml -w fdi-lighthouse-ml -o table
az ml schedule disable -n lighthouse-monthly -g rg-fdi-lighthouse-ml -w fdi-lighthouse-ml
az ml schedule enable  -n lighthouse-monthly -g rg-fdi-lighthouse-ml -w fdi-lighthouse-ml
```

Because the pipeline inputs use `azureml:lh-*@latest`, refreshing the monthly
data is just:

1. Overwrite `lighthouse-raw/demo/accounts.csv` and `drivers.csv` in the
   storage container (or commit + push and let CI do it).
2. Register a new version of the impacted asset:
   ```bash
   az ml data create --name lh-accounts -w fdi-lighthouse-ml -g rg-fdi-lighthouse-ml \
     --type uri_file --path azureml://datastores/lighthouse_raw/paths/demo/accounts.csv
   ```
3. The next scheduled run (or manual trigger) automatically picks the new
   version via the `@latest` alias.

## 2. GitHub Actions

All Actions authenticate via **OIDC federated credentials** — no client
secrets stored in GitHub. Run the bootstrap once to create the AAD app and
federation:

```bash
./infra/scripts/setup-github-oidc.sh <github-org>/<repo> master
```

The script prints three values to paste into
*Settings → Secrets and variables → Actions → Variables*:

- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`

### `docker-build.yml`

Rebuilds and pushes `fdi-lighthouse:py314` to ACR only when
`lh_v2/**`, `docker/**`, or `pyproject.toml` change (or manually via
`workflow_dispatch`). Uses Buildx with a registry-side `buildcache` tag for
fast incremental rebuilds. Also pushes a content-addressed
`py314-<sha>` tag for traceability.

### `aml-train.yml`

`workflow_dispatch`-friendly pipeline submission that:

1. Patches the ACR FQDN into the component.
2. Submits `training/pipeline.yaml`.
3. Streams the job to completion.
4. Downloads the `artifacts/` output and (optionally) republishes
   `portal/index.html` + artifacts to the storage `$web` container.

Also wired as a `schedule: '15 2 1 * *'` safety net — if the AML schedule
is ever disabled, GitHub still runs the monthly forecast 15 minutes later.

## End‑to‑end change propagation

```
lh_v2 change
    │
    ├── GitHub push → docker-build.yml → ACR:fdi-lighthouse:py314 (new digest)
    │
    └── Next schedule tick (AML or GH) →
            pipeline.yaml → component.yaml (image=py314) →
            AmlCompute pulls new image → train_entrypoint.py →
            manifest.json + CSVs + charts → $web/ → live portal
```
