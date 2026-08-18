from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from phase0c.config import ExperimentConfig
from phase0c.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReCoAlign Phase 0-C")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "protocol.json")
    parser.add_argument("--force-behavior", action="store_true")
    parser.add_argument("--force-patch", action="store_true")
    parser.add_argument("--recompute-test", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config, root=ROOT)
    result = run_experiment(
        config,
        force_behavior=args.force_behavior,
        force_patch=args.force_patch,
        recompute_test=args.recompute_test,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
