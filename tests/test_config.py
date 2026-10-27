import pytest

from credit_risk.config import load_config


def test_broken_config_fails_early(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("environment: test\n")
    with pytest.raises(ValueError, match="missing top-level keys"):
        load_config(bad)
