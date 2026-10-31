"""One command for the whole project: data -> product tags -> training -> analyses -> LLM audit -> assistant
evaluation -> figures (incl. the report figures) -> demo.

    python run_pipeline.py                 # everything, about 25 minutes on a 2-core laptop (Excel reading ~2 min)
    python run_pipeline.py --quick         # smoke test: prepare -> tags -> quick train -> analysis
    python run_pipeline.py --sweep --stress    # also the beta/lambda trade-off and the label-bias stress test
    python run_pipeline.py --steps train,figures,demo
    python run_pipeline.py --llm api       # use the LLM API for tags, audit and assistant (needs LLM_* env vars)
    python run_pipeline.py --llm api --steps tags,llmtags,audit,assistant    # only the real-LLM results of the report

Then start the web app:   python -m fairreturn.backend      (http://127.0.0.1:8000)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from fairreturn import config as cfg

ALL_STEPS = ["prepare", "tags", "llmtags", "train", "sweep", "stress", "analysis", "audit", "assistant", "figures", "demo"]
OPTIONAL = ("sweep", "stress", "llmtags")      # only run when asked for (--sweep, --stress, --steps ...)
DEFAULT_STEPS = tuple(step for step in ALL_STEPS if step not in OPTIONAL)
QUICK_DEFAULT_STEPS = ("prepare", "tags", "train", "analysis")
FULL_RUN_ONLY_STEPS = {"audit", "assistant", "figures", "demo"}
STEP_DEPENDENCIES = {
    "tags": {"prepare"},
    "llmtags": {"prepare", "tags"},
    "train": {"prepare"},
    "sweep": {"prepare", "train"},
    "stress": {"prepare", "train"},
    "analysis": {"prepare", "train"},
    "audit": {"prepare", "train"},
    "assistant": {"prepare", "train"},
    "figures": {"prepare", "train", "analysis"},
    "demo": {"prepare", "train"},
}


def resolve_steps(
    requested: list[str] | None,
    *,
    quick: bool,
    sweep: bool,
    stress: bool,
) -> list[str]:
    """Expand an explicit step selection with prerequisites in canonical order."""
    selected = set(QUICK_DEFAULT_STEPS if requested is None and quick else DEFAULT_STEPS if requested is None else requested)
    if sweep:
        selected.add("sweep")
    if stress:
        selected.add("stress")

    unknown = selected - set(ALL_STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {ALL_STEPS}")
    if quick:
        unsupported = selected & FULL_RUN_ONLY_STEPS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                f"--quick writes subsampled results and does not save trained models, "
                f"so it cannot run: {names}. Use the full pipeline without --quick "
                "for report figures and model-dependent steps."
            )

    queue = list(selected)
    while queue:
        step = queue.pop()
        for dependency in STEP_DEPENDENCIES.get(step, set()):
            if dependency not in selected:
                selected.add(dependency)
                queue.append(dependency)
    return [step for step in ALL_STEPS if step in selected]


def run(args: list[str], title: str):
    print(f"\n=== {title} ===\n$ python {' '.join(args)}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, *args], cwd=cfg.ROOT)
    if r.returncode != 0:
        raise SystemExit(f"step '{title}' failed (exit code {r.returncode})")
    print(f"--- {title} done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=None, help=f"comma-separated subset of {ALL_STEPS}")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="include the beta x lambda sweep (~8 min)")
    ap.add_argument("--stress", action="store_true", help="include the injected label-bias stress test (~10 min)")
    ap.add_argument("--llm", choices=["auto", "api", "mock"], default="auto",
                    help="LLM backend for product tags and the audit (auto = API if configured, else mock)")
    ap.add_argument("--llm-tags-in-model", action="store_true",
                    help="rebuild features with the LLM product tags (otherwise keyword-rule tags are used)")
    ap.add_argument("--force-prepare", action="store_true", help="rebuild data/processed even if it exists")
    a = ap.parse_args()

    requested = [step.strip() for step in a.steps.split(",") if step.strip()] if a.steps else None
    try:
        steps = resolve_steps(requested, quick=a.quick, sweep=a.sweep, stress=a.stress)
    except ValueError as exc:
        ap.error(str(exc))
    print(f"Steps: {','.join(steps)}")
    t0 = time.time()

    if "prepare" in steps and (a.force_prepare or not cfg.LINES_FILE.exists()):
        run(["-m", "fairreturn.data.prepare"], "1. prepare data")
    elif "prepare" in steps:
        print("\n=== 1. prepare data === (skipped: data/processed exists; use --force-prepare to rebuild)")
    if "tags" in steps:
        run(["-m", "fairreturn.genai.product_tagger", "--backend", a.llm, "--top", "300" if a.quick else "1000"],
            "2. LLM product tags + evaluation")
        if a.llm_tags_in_model:
            run(["-m", "fairreturn.data.prepare", "--tags", "llm"], "2b. rebuild features with LLM tags")
    if "llmtags" in steps:
        run(["-m", "fairreturn.experiments.run_all", "--product-tags", "llm", "--methods", "gbdt_blind", "gbdt_notags",
             "vib_grl", "--seeds", "3", "--bootstrap", "0", "--quiet"], "2c. value of the LLM product tags (report Table 6)")
    if "train" in steps:
        run(["-m", "fairreturn.experiments.run_all", "--quiet"] + (["--quick"] if a.quick else []),
            "3. train and evaluate all methods")
    if "sweep" in steps:
        run(["-m", "fairreturn.experiments.run_all", "--sweep"] + (["--quick"] if a.quick else []), "3b. beta x lambda sweep")
    if "stress" in steps:
        run(["-m", "fairreturn.experiments.run_all", "--inject-bias", "0.03", "--seeds", "1", "--bootstrap", "0",
             "--methods", "gbdt_blind", "gbdt_reweigh", "mlp", "vib_grl", "--quiet"]
            + (["--quick"] if a.quick else []), "3c. label-bias stress test")
    if "analysis" in steps:
        run(["-m", "fairreturn.experiments.analysis"], "4. extra analyses")
    if "audit" in steps:
        n = ["--n-pos", "10", "--n-neg", "10"] if a.quick else []
        run(["-m", "fairreturn.genai.llm_audit", "--backend", a.llm] + n, "5. LLM audit (E1 + E2)")
    if "assistant" in steps:
        run(["-m", "fairreturn.genai.assistant_eval", "--backend", a.llm, "--n", "20" if a.quick else "100"],
            "5b. decision-assistant evaluation")
    if "figures" in steps:
        run(["-m", "fairreturn.experiments.make_figures"], "6. figures")
        run(["-m", "fairreturn.experiments.report_figures"], "6b. report figures")
    if "demo" in steps:
        run(["-m", "fairreturn.backend.demo_bundle"], "7. demo bundle for the web app")
    elapsed = (time.time() - t0) / 60
    if a.quick:
        print(f"\nQuick pipeline completed in {elapsed:.1f} min.\n"
              f"Quick results: {cfg.RESULTS_DIR / 'quick_main'}\n"
              "No model checkpoints were saved. Run `python run_pipeline.py` without "
              "--quick before starting the web app.")
    else:
        print(f"\nAll done in {elapsed:.1f} min.\n"
              f"Results: {cfg.RESULTS_DIR / 'summary.md'}\nFigures: {cfg.FIG_DIR}\n"
              f"Start the web app:  python -m fairreturn.backend   ->  http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
