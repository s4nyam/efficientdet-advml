# Security

## Reporting

Open an issue for anything non-sensitive. For a credential exposure or
something that should not be public, email the maintainer directly rather than
filing publicly.

## Known exposure in this repository's history

Five notebooks in the 2023 release contained live Roboflow credentials:

- A Roboflow account API key in `2_YOLO8.ipynb`.
- Two dataset download keys embedded in `universe.roboflow.com/ds/...?key=...`
  URLs, across four notebooks.

They are replaced in v2.0.0 with environment-variable placeholders, but
**remain in git history** on any clone or fork made before the rewrite.

### What to do

1. **Rotate the Roboflow API key.** Do this first. Scrubbing files hides the
   string; it does not revoke the credential.
2. Regenerate the dataset download links from the Roboflow UI.
3. Optionally rewrite history with `git filter-repo` — see
   [`MIGRATION.md`](MIGRATION.md). This invalidates existing clones, so weigh it
   against simply rotating.

## Preventing a repeat

Two layers are configured:

- **`gitleaks` as a pre-commit hook** — `pre-commit install` once, then commits
  carrying a credential are blocked locally.
- **`gitleaks` in CI** — `.github/workflows/ci.yml` scans full history on every
  push and pull request.

Plus the `detect-private-key` hook, and `.gitignore` entries for `.env`,
`*.pem` and `secrets.yaml`.

## Handling credentials in notebooks

Never paste a key into a cell. Read it from the environment:

```python
import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
```

On Colab, use the secrets panel:

```python
from google.colab import userdata
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
```

Notebook **outputs** leak too. A cell that prints a dataset URL saves that URL
into the `.ipynb`. Clear outputs before committing, or run `nbstripout`.
