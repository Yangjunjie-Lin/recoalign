# Environment profiles

These files are versioned bootstrap profiles, not claims that every resolved transitive package is
identical across time. Every run captures the actual `pip freeze` output and its SHA-256 digest in
`environment.json`.

- `wsl2-cu126-py310.yml`: local WSL2 profile for the RTX 3060 development machine.
- `cpu-ci-py310.txt`: minimal CPU-only CI installation profile.

Create the WSL2 environment with:

```bash
conda env create -f environments/wsl2-cu126-py310.yml
conda activate recoalign
pip install -e .
```

Before a reportable server experiment, add a separate versioned environment profile rather than
silently reusing the local file.
