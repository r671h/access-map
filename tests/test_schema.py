import pytest
from pydantic import ValidationError

from accessmap.vision.schema import response_model

FEATURE = {
    "type": "raised_curb",
    "box_2d": [600, 100, 700, 400],
    "confidence": 0.8,
    "estimated_distance_m": 6.0,
    "estimated_height_cm": 12,
    "estimated_width_m": None,
    "permanence": "permanent",
    "description_en": "Raised curb at the corner.",
    "description_de": "Hoher Bordstein an der Ecke.",
}


def _frame(**feature_overrides):
    return {
        "image_usable": True,
        "sidewalk_visible": True,
        "sidewalk_surface": "paving_stones",
        "features": [FEATURE | feature_overrides],
    }


def test_valid_response():
    m = response_model(("curb_ramp", "raised_curb"))
    r = m.model_validate(_frame())
    assert r.features[0].type.value == "raised_curb"


def test_inactive_type_rejected():
    m = response_model(("curb_ramp",))
    with pytest.raises(ValidationError):
        m.model_validate(_frame())


@pytest.mark.parametrize("box", [[700, 100, 600, 400], [0, 0, 1001, 10], [1, 2, 3]])
def test_bad_boxes_rejected(box):
    m = response_model(("raised_curb",))
    with pytest.raises(ValidationError):
        m.model_validate(_frame(box_2d=box))


def test_confidence_range():
    m = response_model(("raised_curb",))
    with pytest.raises(ValidationError):
        m.model_validate(_frame(confidence=1.5))
