import shutil
from pathlib import Path
import structurezyme

def test_no_bespoke_weight_downloader_module():
    pkg_dir = Path(structurezyme.__file__).parent
    assert not (pkg_dir / "download_weights.py").exists(), (
        "Weights come from the `squidly` package's own HF download; "
        "do not add a bespoke downloader."
    )

def test_squidly_step_does_not_reference_bundled_pth():
    pkg_dir = Path(structurezyme.__file__).parent
    src = (pkg_dir / "steps" / "squidly_step.py").read_text()
    assert "squidly_final_models" not in src
    assert ".pth" not in src

def test_squidly_cli_available_message_is_documented():
    # The step must fail loudly (clear message) if the CLI is missing,
    # rather than silently trying to load local weights.
    src = (Path(structurezyme.__file__).parent / "steps" / "squidly_step.py").read_text()
    assert "shutil.which(\"squidly\")" in src or "shutil.which('squidly')" in src
