# lighthouse-2026
Internal EY Library for driver based time-series forecasting.

## Installation Guide
### 1. Clone the Repo
**Option 1: Clone via GitHub Desktop.**
- Download GitHub Desktop
    - Mac users: `brew install --cask github`
    - Windows users: `winget install GitHub.GitHubDesktop`
- Open GitHub Desktop, sign in, then find and clone this repo

**Option 2: Clone manually**
- Use Git clone command and clone via HTTPs or SSH method. (SSH recommended, HTTPs may not be stable due to EY proxy.)

HTTPS: `https://github.com/ey-org/lighthouse-2026.git`

SSH: `git@github.com:ey-org/lighthouse-2026.git`

**For Mac users:**
- `brew` (homebrew) is required for installation, follow instructions here: [homebrew install link](https://brew.sh/)

### 2. Install Dependencies
**1. Install `uv`, `python` and `ruff`.**
- **Mac users**
    - Install `uv` and `ruff`: `brew install uv ruff`
    - Install `python`: `uv python install 3.14`
- **Windows users**
    - Install `uv`, `python`, and `ruff`: `winget install astral-sh.uv Python.Python.3.14 astral-sh.ruff`

**For the rest of the installation, you will need to run commands from inside the lighthouse directory (lighthouse-2026).**

**2. Setup python env and install dependencies**
- Run `uv sync` to create virtual environment, and download dependencies.

### 3. Verify application is working
Run `uv run task verify` to run lighthouse testing suite and static type checker.
- Testing suite:
    - If all test cases pass, then the core lighthouse functionality is working properly.
    - If a test case fails, then contact a lighthouse developer.
- Static type checker (pyright)
    - If passes, great!
    - If fails, less important, and core lighthouse functionality still works.