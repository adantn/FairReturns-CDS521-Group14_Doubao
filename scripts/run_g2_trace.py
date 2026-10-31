"""Report-compatible entry point for the G2 counterfactual API audit.

This wrapper keeps the filename referenced in the report and delegates to
`rerun_g2_audit.py`, which sends twin-order prompts to the DeepSeek API and
appends the raw response records to the evidence log.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("rerun_g2_audit.py")
    sys.argv = [str(target), *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
