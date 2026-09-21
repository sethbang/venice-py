"""``camera_trajectory`` constraints, including the two JSON Schema cannot express.

``POST /video/queue`` accepts 2-12 camera poses for H3 Max Multi-Angle. Four
constraints are expressible in the schema (count, ``time`` 0-1, ``elevation``
-90..90, ``distance`` > 0) and two are stated only in the spec's prose:
**strictly increasing time** and **total absolute azimuth travel of at most 32
turns**. Prose constraints have nothing enforcing them, which is exactly why
they are worth testing.
"""

import pytest
from pydantic import ValidationError

from venice_ai.types.api.requests.video import (
    MAX_AZIMUTH_TRAVEL_DEGREES,
    CameraKeyframe,
    VideoImageToVideoRequest,
)


def kf(time: float, azimuth: float, elevation: float = 0.0, distance: float = 1.0) -> dict:
    return {"time": time, "azimuth": azimuth, "elevation": elevation, "distance": distance}


def build(trajectory):
    return VideoImageToVideoRequest(
        model="m",
        prompt="p",
        image_url="https://e.com/a.png",
        duration="5s",
        camera_trajectory=trajectory,
    )


class TestControl:
    def test_request_is_valid_without_a_trajectory(self):
        """Guards the rest of this file: a rejection below is about the
        trajectory, not about the surrounding request being malformed."""
        assert build(None).camera_trajectory is None

    def test_minimal_valid_trajectory(self):
        req = build([kf(0, 0), kf(1, 45)])
        assert len(req.camera_trajectory or []) == 2
        assert isinstance((req.camera_trajectory or [])[0], CameraKeyframe)

    def test_serialises_onto_the_body(self):
        body = build([kf(0, 0), kf(1, 45)]).model_dump(exclude_none=True)
        assert body["camera_trajectory"][1] == {
            "time": 1.0,
            "azimuth": 45.0,
            "elevation": 0.0,
            "distance": 1.0,
        }


class TestSchemaExpressibleConstraints:
    @pytest.mark.parametrize(
        "trajectory",
        [
            pytest.param([kf(0, 0)], id="one-keyframe"),
            pytest.param([kf(i / 13, 0) for i in range(13)], id="thirteen-keyframes"),
            pytest.param([kf(0, 0, elevation=91), kf(1, 0)], id="elevation-above-90"),
            pytest.param([kf(0, 0, elevation=-91), kf(1, 0)], id="elevation-below-minus-90"),
            pytest.param([kf(0, 0, distance=0), kf(1, 0)], id="distance-zero"),
            pytest.param([kf(0, 0), kf(1.5, 0)], id="time-above-one"),
            pytest.param([kf(-0.1, 0), kf(1, 0)], id="time-below-zero"),
            pytest.param([{"time": 0, "elevation": 0, "distance": 1}, kf(1, 0)], id="no-azimuth"),
            pytest.param([{**kf(0, 0), "zoom": 2}, kf(1, 0)], id="unknown-key"),
        ],
    )
    def test_rejected(self, trajectory):
        with pytest.raises(ValidationError):
            build(trajectory)


class TestProseOnlyConstraints:
    """Neither of these is in the JSON Schema; only this validator catches them."""

    @pytest.mark.parametrize(
        "trajectory",
        [
            pytest.param([kf(0, 0), kf(0, 45)], id="equal-times"),
            pytest.param([kf(0.5, 0), kf(0.2, 45)], id="decreasing-times"),
            pytest.param([kf(0, 0), kf(0.5, 1), kf(0.5, 2)], id="equal-times-late"),
        ],
    )
    def test_time_must_strictly_increase(self, trajectory):
        with pytest.raises(ValidationError, match="strictly increasing"):
            build(trajectory)

    def test_azimuth_travel_at_the_limit_is_allowed(self):
        build([kf(0, 0), kf(1, MAX_AZIMUTH_TRAVEL_DEGREES)])

    def test_azimuth_travel_beyond_the_limit_is_rejected(self):
        with pytest.raises(ValidationError, match="32 turns"):
            build([kf(0, 0), kf(1, MAX_AZIMUTH_TRAVEL_DEGREES + 1)])

    def test_travel_is_cumulative_not_endpoint_distance(self):
        """Back-and-forth motion accumulates; end-to-start would read as zero."""
        half = MAX_AZIMUTH_TRAVEL_DEGREES
        with pytest.raises(ValidationError, match="32 turns"):
            build([kf(0, 0), kf(0.5, half), kf(1, 0)])

    def test_travel_uses_absolute_values(self):
        """Negative legs must not cancel positive ones."""
        leg = MAX_AZIMUTH_TRAVEL_DEGREES
        with pytest.raises(ValidationError, match="32 turns"):
            build([kf(0, 0), kf(0.5, -leg), kf(1, 0)])
