# Third-Party Notices and License Boundary

This is an inventory and boundary statement, not a relicensing instrument. Versions and exact
runtime provenance are recorded where frozen manifests provide them; current installation
constraints are in `pyproject.toml` and reproducibility files.

| Dependency or asset | Role | Included in release | License/terms responsibility |
| --- | --- | --- | --- |
| LLaVA / `liuhaotian/llava-v1.5-7b` | Frozen VLM backbone | Identifier and metadata only; no weights | Upstream model card, code license, and weight terms |
| PyTorch | Tensor/runtime library | Dependency metadata only | PyTorch upstream license |
| Transformers | Model loading/runtime | Dependency metadata only | Hugging Face upstream license |
| bitsandbytes | Quantized runtime | Dependency metadata only | Upstream project license and platform terms |
| OpenCLIP | Historical diagnostic dependency | Dependency metadata only; no weights | Upstream project and model-weight terms |
| Benchmark datasets | Historical/evaluation inputs | No restricted images or gated annotations | Each dataset's license, access, and citation terms |
| Synthetic rendering dependencies | Synthetic-world generation | Code/configuration only unless tracked lightweight assets are original | Each upstream software license |
| NumPy, Pillow, PyYAML, jsonschema, tqdm | Core software dependencies | Dependency metadata only | Respective upstream licenses |

Repository-authored source and documentation are distributed under the repository's Apache-2.0
license. That license must not be interpreted as granting rights in any third-party checkpoint,
dataset, benchmark image, annotation, or trademark.
