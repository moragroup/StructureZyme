# tests/test_cli_init_multi_substrate.py
"""`structurezyme init` must write a config that surfaces `multi_substrate_mode`
so users discover the flag. RunConfig serializes the field automatically via
model_dump(); this test pins that contract so a future serialization change
cannot silently drop the flag from the init template."""
import yaml

from structurezyme.cli import _template


def test_init_template_includes_multi_substrate_mode(tmp_path):
    out = tmp_path / "run.yml"
    _template().write(out)
    data = yaml.safe_load(out.read_text())
    assert data["multi_substrate_mode"] == "off"


def test_init_template_default_is_off(tmp_path):
    """The template must default to 'off' (feature opt-in), not enable
    multi-substrate behavior implicitly for existing users."""
    out = tmp_path / "run.yml"
    _template().write(out)
    data = yaml.safe_load(out.read_text())
    assert data["multi_substrate_mode"] in {"off", "separate", "together"}
    assert data["multi_substrate_mode"] == "off"
