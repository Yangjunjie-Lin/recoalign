"""Allow ``python -m recoalign`` to invoke the CLI."""

from recoalign.cli import main

raise SystemExit(main())
