# Fair Returns

Fair Returns is the Group 14 submission for CDS521 Foundation of Artificial
Intelligence. It studies country-invariant return-risk
prediction for cross-border e-commerce with a variational information
bottleneck and a gradient-reversal adversary (VIB + GRL).

## Repository

Source code, report, deterministic metrics, and audit artifacts are available
at [github.com/adantn/FairReturns-CDS521-Group14_Doubao](https://github.com/adantn/FairReturns-CDS521-Group14_Doubao).

## Headline result

On the temporal test split of Online Retail II, VIB + GRL achieves AUC 0.743
and an equalized-odds false-positive gap of 0.057, averaged over three fixed
random seeds. Delivery address and postage are removed from its inputs, while
a 16-dimensional representation is trained to hide country information from
an equalized-odds adversary.

## Repository layout

```text
fairreturn/                 Python package: data, models, evaluation, GenAI, API
frontend/                   Static web interface served by the Flask backend
tests/                      Unit, integration, API, and numerical checks
scripts/                    Deterministic metric and G2 audit verification
deliverables/
  report/                   Final DOCX/PDF and revision log
  data/                     Deterministic summary, seed, and group metrics
  evidence/                 G2 raw API trace and sanitized audit input
  presentation/             Slides and demo video
run_pipeline.py             End-to-end preparation, training, and analysis
pyproject.toml              Installable package metadata
requirements.txt            Core runtime dependencies
```

The public source workbooks under `data/raw/` and the evidence under
`deliverables/` are committed. Generated directories (`data/interim/`,
`data/processed/`, and `outputs/`) and the local archive `origin/` are excluded
from version control.

## Setup

Python 3.10 or newer is required.

```powershell
python -m pip install -e .
```

## Verify the reported metrics

This check is offline and does not need a model checkpoint:

```powershell
python scripts\verify_metrics.py
# Equivalent to the filename used in the report:
python scripts\eval_deterministic.py
```

It recomputes every mean and standard deviation in `summary.csv` from the
individual seed runs, checks the six country groups per method, and verifies
the seed-0 demographic parity gap against `group_metrics.csv`.

`eval_deterministic.py` is a thin wrapper around `verify_metrics.py`, so both
names run the same checks.

## Rerun the G2 audit

The sanitized input and recorded output are in `deliverables/evidence/`.
Configure the API key only in the current environment:

```powershell
$env:DEEPSEEK_API_KEY = "<your-key>"
python scripts\rerun_g2_audit.py
# Equivalent to the filename used in the report:
python scripts\run_g2_trace.py
```

By default, the script reads `audit_source_results.csv` and writes a new trace
to `outputs/g2_audit_raw_trace.jsonl`. Use `--source-csv`, `--output`,
`--condition`, or `--limit` to override those defaults.

`run_g2_trace.py` is a thin wrapper around `rerun_g2_audit.py`, so both names
run the same audit.

The committed `g2_audit_raw_trace.jsonl` contains 200 records for 100
counterfactual twin pairs. The requested alias was `deepseek-chat`, and the API
reported `deepseek-flash` as the resolved model at temperature 0.0.

## Run the full research pipeline

The two UCI Online Retail II workbooks are included in `data/raw/` under the
[CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/). The dataset
is available from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
To rerun the published pipeline from the included source data:

```powershell
python run_pipeline.py
```

For a dependency-aware smoke test:

```powershell
python run_pipeline.py --quick
```

Quick mode runs `prepare`, `tags`, `train`, and `analysis` using a subsampled
training set. Its quick model run intentionally does not save model checkpoints
or report figure outputs, so it does not run `audit`, `assistant`, `figures`,
or `demo`. Run the full pipeline without `--quick` before starting the web
application or reproducing those outputs.

Useful component commands:

```powershell
python -m fairreturn.data.prepare
python -m fairreturn.experiments.run_all --quick
python -m fairreturn.backend
python -m tests.selftest
```

The web application runs at `http://127.0.0.1:8000` after the trained model
bundle has been generated.

## Design Rationale

Standalone adversarial training (GRL-only) suffers from severe minimax instability in discrete, high-dimensional neural representations, causing latent collapse and sharp accuracy drop (AUC down to 0.693). Conversely, pure information compression (VIB-only) inherently retains geographic shortcuts due to empirical cross-border return-rate disparities. Combining them creates a mutual stabilizing effect: VIB's Gaussian reparameterization and KL regularizer inject continuous smoothness into the latent space to pacify adversarial gradient oscillations, while GRL acts as a surgical discriminator to precisely excise sensitive country-aligned dimensions.


## Configuration and security

LLM components read `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` from the
environment. The G2 rerun uses `DEEPSEEK_API_KEY`. No credentials are stored in
the repository; `.env`, key files, token files, and generated outputs are
ignored.
