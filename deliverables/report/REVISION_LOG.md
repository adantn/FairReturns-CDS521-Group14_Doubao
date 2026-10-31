# Fair Returns final report revision log

Revision date: 2026-10-31

Final report: `FairReturns_Complete_Report.docx` and `FairReturns_Complete_Report.pdf`

Verification scripts: `../../scripts`

## 1. Deliverable status

- The final report is exported as both DOCX and PDF.
- The PDF contains 26 pages, including the main text, references, and appendices.
- Figures 1-7 and Appendix C figures C1-C5 are present with their captions.
- Tables 5, 6, and 7 are complete, with no split rows or truncated cells.
- The submitted DOCX and PDF contain no draft callout, unresolved team notes,
  yellow placeholder markers, or highlight remnants.

## 2. Deterministic result verification

`scripts/verify_metrics.py` recomputes every reported mean and standard
deviation from `deliverables/data/seeds.csv` and compares the values with
`deliverables/data/summary.csv`. It also verifies that `group_metrics.csv`
contains the six country groups for each method and that the seed-0 demographic
parity gap matches the corresponding seed-0 value.

The verification covers:

- Test AUC
- Brier score
- Demographic parity gap
- Equalized-odds false-positive-rate gap

All recomputed values match the summary file within numerical tolerance.

## 3. G2 audit evidence

`deliverables/evidence/g2_audit_raw_trace.jsonl` contains 200 JSONL records,
all tagged `run=g2_v2`. The records correspond to 100 counterfactual twin pairs,
with one original and one twin per pair under the plain prompt condition.

Verified values:

| Item | Value |
|---|---|
| Twin pairs | 100 |
| API calls | 200 |
| Prompt condition | Plain |
| Requested model alias | `deepseek-chat` |
| Resolved response model | `deepseek-flash` |
| Finish reason | `stop` for 200/200 calls |
| Decision flip rate at threshold 30 | 0.00% |
| Mean risk delta (twin minus original) | +0.59 points |
| Foreign surcharge impact | +0.97 points |
| Non-UK premium share | 32.79% |
| AUC on original lines | 0.604 |

## 4. Correction history

- The abstract and main results use the verified model metrics rather than
  unqualified relative improvement claims.
- Table 5 reports the G1 product-attribute evaluation as a double-blind human
  verification template. The baseline tags in the deployed system are produced
  by a deterministic rule cache for throughput.
- Table 7 reports the G2 plain-condition values recomputed from the raw API
  trace. Unmeasured G2 conditions are shown as unavailable rather than filled
  with placeholder numbers.
- Appendix A states the team's ownership of the VIB-GRL implementation and
  describes the external DeepSeek API and writing-support tools without naming
  them as authors of the algorithm or the experimental results.

## 5. Remaining limitations

The G2 audit is a single-model, single-prompt, temperature-0 study on 100 twin
pairs. The hard-threshold flip rate of 0.00% should be interpreted together
with the +0.59 point mean risk delta and the +0.97 point foreign surcharge,
which show residual micro-level country sensitivity.
