"""Tests for pipeline step selection and prerequisite expansion."""
from __future__ import annotations

from run_pipeline import DEFAULT_STEPS, QUICK_DEFAULT_STEPS, resolve_steps


def test_quick_default_avoids_model_dependent_steps():
    assert resolve_steps(None, quick=True, sweep=False, stress=False) == list(QUICK_DEFAULT_STEPS)


def test_full_default_includes_model_dependent_steps():
    assert resolve_steps(None, quick=False, sweep=False, stress=False) == list(DEFAULT_STEPS)


def test_explicit_steps_gain_prerequisites_in_order():
    assert resolve_steps(["figures"], quick=False, sweep=False, stress=False) == [
        "prepare",
        "train",
        "analysis",
        "figures",
    ]


def test_sweep_and_stress_expand_from_train_only_selection():
    assert resolve_steps(["train"], quick=False, sweep=True, stress=True) == [
        "prepare",
        "train",
        "sweep",
        "stress",
    ]


def test_quick_rejects_steps_that_need_saved_models():
    try:
        resolve_steps(["audit", "figures"], quick=True, sweep=False, stress=False)
    except ValueError as exc:
        assert "--quick writes subsampled results" in str(exc)
        assert "audit, figures" in str(exc)
    else:
        raise AssertionError("quick mode accepted a saved-model-dependent step")
