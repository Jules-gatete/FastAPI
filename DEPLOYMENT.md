DigitalOcean / Generic Platform deployment notes
===============================================

This file documents a minimal, ready-to-push structure and steps to deploy the FastAPI app to DigitalOcean App Platform (or similar PaaS).

What this repo now contains (deployment-ready):

- `requirements.txt`          # Python deps used by the buildpack
- `runtime.txt`               # Python runtime: set to `python-3.11.9`
- `Procfile`                  # How to start the app (uses $PORT)
- `api.py`                    # FastAPI app (reads $PORT from env)
- `models/` (placeholder)     # `models/.gitkeep` ensures folder exists in repo
- `.gitattributes`            # Tracks model binaries with Git LFS (if used)

Quick checklist before deploy
-----------------------------
1. Ensure `models/` contains your trained model files (or a script to download them at startup).
   - If your models are large, use Git LFS (see below) or host models on external storage (S3, Hugging Face Hub, etc.) and download in startup.
2. Confirm `requirements.txt` contains all required packages and compatible pins for Python 3.11.
3. Commit and push your repo to a Git provider (GitHub/GitLab).

Git LFS (recommended for large model binaries)
----------------------------------------------
If you store large files (model weights, tokenizers, .safetensors, .pt, .bin, .npy), use Git LFS:

1. Install and initialize LFS locally:

   ```bash
   # macOS / Linux
   git lfs install

   # Windows (PowerShell / Admin)
   git lfs install
   ```

2. If `.gitattributes` is present (this repo includes one), commit it. If not, add patterns you want to track, e.g.:

   ```text
   models/** filter=lfs diff=lfs merge=lfs -text
   *.safetensors filter=lfs diff=lfs merge=lfs -text
   *.pt filter=lfs diff=lfs merge=lfs -text
   *.bin filter=lfs diff=lfs merge=lfs -text
   ````

3. Track/commit and push large files normally. New large files matching patterns will be stored in LFS.

Note about already-pushed large files
------------------------------------
If you've already pushed large model files to the repo, and you want them moved into LFS, you must rewrite history (careful — coordinate with collaborators):

```bash
# migrate matching patterns into LFS (rewrites history)
git lfs migrate import --include="models/**,*.safetensors,*.pt,*.bin"
# force-push rewritten history
git push --force origin main
```

This step is optional and disruptive for collaborators.

How to deploy on DigitalOcean App Platform
------------------------------------------
1. Push your repo to GitHub (or a supported provider).
2. In DigitalOcean, create a new App and connect your repo.
3. Choose the `main` branch (or preferred branch) and configure build & run settings:
   - Build command: leave default (the Python buildpack will run `pip install -r requirements.txt`).
   - Run command: leave blank if you have a `Procfile`, or set it to `uvicorn api:app --host 0.0.0.0 --port $PORT`.
4. Ensure the environment variable `PORT` is present — DigitalOcean sets this automatically for App Platform.
5. Deploy and watch logs.

Local testing (before pushing)
------------------------------
Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the app locally:

```powershell
# Runs on port 8000 locally
uvicorn api:app --reload
# or
python api.py
```

Visit `http://127.0.0.1:8000/docs` to inspect endpoints.

Health checks
-------------
The app exposes `/health`. Make sure it returns `status: healthy` before routing traffic.

If `/health` reports `Models directory not found`, ensure `models/` contains the trained model files or a startup script downloads them.

Optional: automatic model download at startup
--------------------------------------------
If you prefer not to commit models, add a small startup script `scripts/download_models.py` which fetches weights from an object store. Call it from a startup hook or from application code on first request.

That's it — with the `Procfile`, `runtime.txt`, and `requirements.txt` set, your repo is ready for DigitalOcean deployment.
