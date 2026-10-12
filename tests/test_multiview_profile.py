import pytest
from fastapi.testclient import TestClient
from spatialmemory import app


def test_multiview_ui_profile_uses_connected_pipeline_stages():
    config = app.upload_config("vggt", True, False, True)
    assert config.association_mode == "joint"
    assert config.multiview_geometry_enabled and config.geometry_supported_views_enabled
    assert (
        config.multiscale_detection_enabled and config.semantic_label_validation_enabled
    )
    assert config.neural_preprocessing == "pad"
    assert config.detection_threshold == 0.35


def test_invalid_ui_profile_is_rejected_before_upload():
    with pytest.raises(ValueError):
        app.upload_config("sparse", True, False, True)
    client = TestClient(app.app)
    result = client.post(
        "/jobs",
        data={
            "geometry_backend": "sparse",
            "semantics_enabled": "true",
            "multiview_object_analysis": "true",
        },
        files={
            "before": ("before.mp4", b"test", "video/mp4"),
            "after": ("after.mp4", b"test", "video/mp4"),
        },
    )
    assert result.status_code == 422
    assert "requires neural" in result.json()["detail"]
