# Data directory

Raw datasets, generated annotations, caches, and extracted embeddings must not be committed.

Recommended local layout:

```text
data/
├── raw/
│   ├── flickr30k/
│   ├── coco/
│   ├── sugarcrepe/
│   ├── aro/
│   └── winoground/
├── processed/
│   └── <dataset>/<version>/
└── cache/
```

Each processed dataset version should include a manifest recording:

- source and download date;
- upstream version or commit;
- split definition;
- sample counts;
- transformation script and arguments;
- content hashes where practical;
- applicable license or access restrictions.

Do not redistribute datasets unless their licenses explicitly permit it.
