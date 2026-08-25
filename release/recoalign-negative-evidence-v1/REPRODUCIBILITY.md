# Reproducibility Protocol

This protocol verifies the released records without loading model weights, running inference,
downloading restricted data, or producing new scientific metrics.

```bash
git clone https://github.com/Yangjunjie-Lin/recoalign.git
cd recoalign
git checkout recoalign-negative-evidence-v1
python -m venv .venv-release-check
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
ruff check .
python -m recoalign validate-research
python -m recoalign validate-paper-package
python scripts/verify_negative_evidence_release.py
```

The verifier checks:

- 14 claims and 0 pending claims;
- `TERMINATE_CURRENT_PROGRAM` and `STOP` selection;
- existence and SHA-256 integrity of frozen manifest entries;
- readability of compressed prediction files;
- release-manifest file hashes; and
- deterministic archive checksums when the archives are present.

Expected verification does not require a GPU. The developer dependency set includes Torch because
historical engineering tests import trainable-interface modules; this is a software test
dependency, not permission to train or infer.
