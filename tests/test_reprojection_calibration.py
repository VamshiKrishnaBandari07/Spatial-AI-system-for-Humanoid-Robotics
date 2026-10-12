import numpy as np
import pytest
from spatialmemory.reprojection import refine_point
from spatialmemory.detector_calibration import DetectorTemperature
from spatialmemory.geometry import project
from spatialmemory.inputs import Manifest


def test_robust_refinement_recovers_point_without_moving_cameras():
    k = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [0, -.4, .4]
    truth = np.array([.1, -.1, 3.])
    pixels = np.array([project(truth[None], k, p)[0][0] for p in poses])
    fit = refine_point(truth + [.2, .1, .5], pixels, np.repeat(k[None], 3, axis=0), poses)
    assert fit.accepted
    assert fit.errors_px.max() < 1e-6
    assert fit.initial_errors_px.max() > 2
    np.testing.assert_allclose(fit.point, truth, atol=1e-7)


def test_uncertainty_cannot_hide_inconsistent_view():
    k = np.eye(3)
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [0, -.4, .4]
    pixels = np.array([[0., 0], [-.1, 0], [.1, 30]])
    fit = refine_point(np.array([0., 0, 4]), pixels, np.repeat(k[None], 3, axis=0),
                       poses, sigma_px=np.array([1., 1, 1000]))
    assert not fit.accepted


def test_reprojection_alone_cannot_accept_zero_baseline():
    fit = refine_point(np.array([0., 0, 3]), np.zeros((2, 2)),
                       np.repeat(np.eye(3)[None], 2, axis=0),
                       np.repeat(np.eye(4)[None], 2, axis=0))
    assert not fit.accepted
    assert fit.covariance is None


def test_temperature_requires_labels_and_detector_provenance():
    with pytest.raises(ValueError):
        DetectorTemperature.fit(np.ones(3), np.ones(3), "validation", "revision")
    fitted = DetectorTemperature.fit(np.array([-5., -4, 4, 5]),
                                     np.array([0., 1, 0, 1]), "labelled-scenes", "rev")
    assert fitted.temperature > 1
    probabilities = fitted.probabilities(np.array([-2., 2]), "rev")
    assert 0 < probabilities[0] < probabilities[1] < 1
    with pytest.raises(ValueError, match="revision"):
        fitted.probabilities(np.ones(1), "different")


def test_physical_chronology_uses_timezones_not_filenames():
    values = dict(scene_id="test", before=dict(video="z.mp4", recorded_at="2026-10-12T10:00:00Z"),
                  after=dict(video="a.mp4", recorded_at="2026-10-12T12:00:00+01:00"))
    assert Manifest.model_validate(values).before.video == "z.mp4"
    values["after"]["recorded_at"] = "2026-10-12T10:30:00+01:00"
    with pytest.raises(ValueError, match="physically later"):
        Manifest.model_validate(values)
    values["after"]["recorded_at"] = "2026-10-12T12:00:00"
    with pytest.raises(ValueError, match="timezone"):
        Manifest.model_validate(values)
