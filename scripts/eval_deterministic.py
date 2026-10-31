"""Report-compatible entry point for the deterministic metric verifier.

This wrapper keeps the filename referenced in the report and delegates to
`verify_metrics.py`, which recomputes reported means, standard deviations and
group-level parity gaps from the deliverable CSV files.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("verify_metrics.py")
    sys.argv = [str(target), *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
