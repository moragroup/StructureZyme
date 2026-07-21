# tests/test_step_runners_smoke.py
from structurezyme import registry


def test_every_step_has_a_runner_attached():
    for name, spec in registry.STEPS.items():
        assert callable(spec.runner), f"{name} has no runner wired"
