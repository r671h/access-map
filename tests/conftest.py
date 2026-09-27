from pathlib import Path

import pytest
import yaml

PROJECT = {
    "goal": "real_users",
    "autonomy": "checkpoints",
    "area": {"name": "Test area", "bbox": [8.526, 52.016, 8.544, 52.030]},
    "profiles": ["wheelchair"],
    "detection": {"types": ["curb_ramp", "raised_curb", "stairs"]},
}


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "project.yaml").write_text(yaml.safe_dump(PROJECT), encoding="utf-8")
    monkeypatch.setenv("ACCESSMAP_ROOT", str(tmp_path))
    # Tests must never see real keys.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MAPILLARY_TOKEN", raising=False)
    return tmp_path


@pytest.fixture
def settings(project_root):
    from accessmap.config import Settings

    return Settings(project_root)
