# Maintenance Policy

```yaml
active_method_development: false
new_scientific_experiments: false
security_fixes_allowed: true
documentation_fixes_allowed: true
artifact_integrity_fixes_allowed: true
scientific_result_changes_allowed: false
new_claims_allowed: false
```

Permitted future changes are broken-link repairs, typographical corrections, security fixes,
dependency-metadata fixes, artifact-verifier fixes, and documentation clarifications that do not
change scientific meaning.

Prohibited changes are new hypotheses, experiments, models, benchmarks, methods, reinterpretation
of frozen evidence, weakened gates, renewed paper roadmaps, prediction or metric changes, and any
attempt to reactivate the ReCoAlign method line.

Scientific artifacts must remain byte-identical to the evidence-freeze tag. Corrections to release
metadata must identify the affected release file, retain prior checksums, and use a new patch-level
resource release rather than moving an existing tag.
