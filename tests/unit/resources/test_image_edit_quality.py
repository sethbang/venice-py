"""``quality`` must never reach ``POST /image/edit``.

The endpoint declares ``additionalProperties: false`` and has no ``quality``
field, so the server rejects the *whole request* with HTTP 400
(``Unrecognized key(s) in object: 'quality'``) rather than ignoring it. The
same request without it returns 200. That makes sending it unconditionally
fatal, not merely wasteful — which is why the parameter is dropped rather than
forwarded, and why it is worth a test rather than a comment.

``quality`` *is* valid on ``POST /image/multi-edit``; that path is covered
separately and must keep working.
"""

from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.image import Image
from venice_ai.types.api.requests.images import ImageEditRequest, ImageMultiEditRequest


class TestEditRequestModel:
    def test_edit_request_has_no_quality_field(self):
        """Modelling it at all invites it back onto the wire."""
        assert "quality" not in ImageEditRequest.model_fields

    def test_edit_request_body_never_carries_quality(self):
        body = ImageEditRequest(prompt="p", image="https://e.com/a.png").model_dump(
            exclude_none=True
        )
        assert "quality" not in body

    def test_multi_edit_request_still_models_quality(self):
        """The field is legitimate on the multi-edit endpoint."""
        assert "quality" in ImageMultiEditRequest.model_fields


@pytest.fixture
def image_resource() -> Image:
    client = Mock()
    client._request = AsyncMock(return_value=b"img-bytes")
    return Image(client)


class TestEditDropsQuality:
    @pytest.mark.asyncio
    async def test_edit_does_not_send_quality(self, image_resource: Image):
        with pytest.warns(DeprecationWarning, match="quality is not accepted"):
            await image_resource.edit(prompt="p", image="https://e.com/a.png", quality="medium")
        payload = cast(Any, image_resource._client._request).call_args.kwargs["json_data"]
        assert "quality" not in payload

    @pytest.mark.asyncio
    async def test_edit_still_sends_everything_else(self, image_resource: Image):
        """Dropping one key must not drop its neighbours."""
        with pytest.warns(DeprecationWarning):
            await image_resource.edit(
                prompt="p",
                image="https://e.com/a.png",
                quality="high",
                output_format="png",
                resolution="2K",
            )
        payload = cast(Any, image_resource._client._request).call_args.kwargs["json_data"]
        assert payload["output_format"] == "png"
        assert payload["resolution"] == "2K"
        assert payload["prompt"] == "p"

    @pytest.mark.asyncio
    async def test_no_warning_when_quality_is_not_passed(self, image_resource: Image):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await image_resource.edit(prompt="p", image="https://e.com/a.png")
