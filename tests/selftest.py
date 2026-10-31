"""Run every test without needing pytest:   python -m tests.selftest

(pytest also works:  python -m pytest tests -q)
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

MODULES = ["tests.test_vib_grl", "tests.test_pipeline", "tests.test_metrics", "tests.test_genai", "tests.test_report",
           "tests.test_run_pipeline", "tests.test_torch_parity", "tests.test_api"]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    passed = failed = skipped = 0
    t_all = time.time()
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t0 = time.time()
            try:
                out = getattr(mod, name)()
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {mod_name}.{name}")
                traceback.print_exc()
                continue
            if isinstance(out, str) and out.startswith("skipped"):
                skipped += 1
                print(f"[SKIP] {mod_name}.{name}: {out}")
            else:
                passed += 1
                print(f"[PASS] {mod_name}.{name} ({time.time() - t0:.1f}s)")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped in {time.time() - t_all:.0f}s")
    print("ALL CHECKS PASSED" if failed == 0 else "SOME CHECKS FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
