# Docker image (Python 3.14) for Azure ML

The image installs **`lh_v2` from the repo** using the locked dependencies in [`../uv.lock`](../uv.lock) and [`uv sync`](https://github.com/astral-sh/uv) into `/opt/lh-v2/.venv`. Use it as the **command job environment** in Azure Machine Learning, pulling from ACR (see Bicep outputs for `acrLoginServer` and `acrNameOut`).

## Build locally (sanity)

From the **repository root**:

```bash
docker build -f docker/Dockerfile -t fdi-lighthouse:py314 .
docker run --rm fdi-lighthouse:py314
```

## Build in Azure Container Registry (recommended for ML jobs)

1. Get your registry name: `az acr list -g rg-fdi-lighthouse-ml -o table` (or the name from the Bicep deployment output `acrNameOut`).

2. From the **repository root**:

```bash
export ACR_NAME="<your-acr-name>"

az acr build -r $ACR_NAME -f docker/Dockerfile -t fdi-lighthouse:py314 . --platform linux/amd64
```

Use `--platform linux/amd64` for consistency on Apple Silicon. The AmlCompute cluster in Azure is **Linux AMD64** by default.

3. (Optional) Tag with the run id or git SHA in CI:

```bash
az acr build -r $ACR_NAME -f docker/Dockerfile -t fdi-lighthouse:py314 -t fdi-lighthouse:$(git rev-parse --short HEAD) .
```

## ACR login (when `adminUserEnabled` is true in Bicep)

```bash
az acr login -n $ACR_NAME
```

## Smoke job on Azure ML (CLI v2)

1. `az extension add -n ml -y`
2. Set your resource group, workspace, and the full image. Replace `ACR_NAME_PLACEHOLDER` in [`smoke-job.yaml`](smoke-job.yaml) with the registry host **without** `https://` (Bicep output: `acrLoginServer`, e.g. `myacrx.azurecr.io`).

Then run:

```bash
export RG="rg-fdi-lighthouse-ml"
export WS="fdi-lighthouse-ml"

az ml job create -f smoke-job.yaml --resource-group "$RG" --workspace-name "$WS"
```

Or use `sed` to fill the image, then:

```bash
az ml job create -f smoke-job.yaml --resource-group "$RG" --workspace-name "$WS"
```

3. The job should **Complete**; see logs in **Azure Machine Learning studio** (Jobs). If the cluster was scaled to zero, the first run can take a few minutes to start.

## Custom training command (later)

Point the job `command` to a small Python **entry** script (mount `inputs` / `config.yml` from a job input or code snapshot). See [../docs/azure-ml-data-and-jobs.md](../docs/azure-ml-data-and-jobs.md).

## Troubleshooting

- **Build failure on a wheel for Linux:** Some third-party wheels for Python 3.14 may be missing. Relax a pin, use a different version, or open an issue. Building on `linux/amd64` matches the cloud.
- **`AcrAuthenticationError` in the job:** Confirm the Bicep deployment run finished and the workspace managed identity has **AcrPull** (this repo’s template adds it to the ACR you linked). Wait a few minutes and retry the job.
- **Memory:** The full dependency set (Prophet, XGBoost, etc.) is heavy; for large jobs, pick a larger `vmSize` in Bicep or a separate **high-memory** cluster in the same workspace.
