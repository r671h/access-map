from pathlib import Path

import pytest
from pydantic import ValidationError

from accessmap.config import ProjectConfig, Settings, load_project

REPO = Path(__file__).resolve().parents[1]


def test_repo_project_yaml_is_valid():
    cfg = load_project(REPO / "config" / "project.yaml")
    assert cfg.area.bbox[0] < cfg.area.bbox[2]
    assert cfg.budget.max_gemini_calls > 0


def test_settings_reads_project_and_env_file(project_root, monkeypatch):
    (project_root / ".env").write_text("GEMINI_API_KEY=abc\nMAPILLARY_TOKEN=MLY|x\n")
    s = Settings(project_root)
    assert s.project.area.name == "Test area"
    assert s.secrets.gemini_api_key == "abc"
    assert s.secrets.mapillary_token == "MLY|x"
    assert s.paths.processed == project_root / "data" / "processed"


def test_missing_keys_are_none(settings):
    assert settings.secrets.gemini_api_key is None
    assert settings.secrets.mapillary_token is None


def test_bbox_order_is_validated():
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"area": {"name": "x", "bbox": [8.5, 52.0, 8.4, 52.1]}})


def test_unknown_barrier_type_rejected():
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(
            {"area": {"name": "x", "bbox": [8.4, 52.0, 8.5, 52.1]},
             "detection": {"types": ["curb_ramp", "lava"]}}
        )
