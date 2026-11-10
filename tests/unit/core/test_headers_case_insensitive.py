"""Response headers on VeniceBaseModel are looked up without regard to case.

aiohttp hands the SDK a ``CIMultiDictProxy`` that keeps the server's casing, and
Venice sends ``CF-RAY`` and ``X-RateLimit-Remaining`` in title case.
"""

import json
from types import SimpleNamespace

import pytest
from multidict import CIMultiDict, CIMultiDictProxy

from venice_ai.core.models.base import VeniceBaseModel


class _Model(VeniceBaseModel):
    pass


def _model_with(headers: dict[str, str]) -> _Model:
    model = _Model()
    model._response = SimpleNamespace(headers=CIMultiDictProxy(CIMultiDict(headers)))
    return model


def _spellings(name: str) -> list[str]:
    return [name.lower(), name.upper(), "-".join(p.title() for p in name.split("-"))]


@pytest.mark.parametrize("name", _spellings("cf-ray"))
def test_request_id_reads_any_casing(name: str) -> None:
    assert _model_with({name: "abc123-LAX"}).request_id == "abc123-LAX"


@pytest.mark.parametrize("name", _spellings("x-venice-version"))
def test_venice_version_reads_any_casing(name: str) -> None:
    assert _model_with({name: "20261105.1"}).venice_version == "20261105.1"


@pytest.mark.parametrize("name", _spellings("x-balance-remaining"))
def test_x402_balance_remaining_reads_any_casing(name: str) -> None:
    assert _model_with({name: "4.5"}).x402_balance_remaining == 4.5


@pytest.mark.parametrize("name", _spellings("x-ratelimit-remaining-requests"))
def test_response_rate_limits_reads_any_casing(name: str) -> None:
    limits = _model_with({name: "17"}).response_rate_limits
    assert limits is not None
    assert limits.remaining_requests == 17


def test_pagination_info_reads_title_case() -> None:
    info = _model_with(
        {
            "X-Pagination-Limit": "10",
            "X-Pagination-Page": "2",
            "X-Pagination-Total": "25",
            "X-Pagination-Total-Pages": "3",
        }
    ).pagination_info
    assert info is not None
    assert (info.limit, info.page, info.total, info.total_pages) == (10, 2, 25, 3)


def test_deprecation_info_reads_title_case() -> None:
    info = _model_with({"X-Venice-Model-Deprecation-Warning": "going away"}).deprecation_info
    assert info is not None
    assert info.warning == "going away"


def test_balance_info_reads_title_case() -> None:
    info = _model_with({"X-Venice-Balance-USD": "12.5", "X-VENICE-BALANCE-DIEM": "3"}).balance_info
    assert info is not None
    assert (info.usd, info.diem) == (12.5, 3.0)


def test_content_safety_info_reads_title_case() -> None:
    info = _model_with({"X-Venice-Is-Blurred": "true"}).content_safety_info
    assert info is not None
    assert info.is_blurred is True


def test_model_info_reads_title_case() -> None:
    info = _model_with({"X-Venice-Model-Id": "stand-in-model"}).model_info
    assert info is not None
    assert info.model_id == "stand-in-model"


class TestHeadersStayADict:
    """``response.headers`` remains a dict for callers, whatever casing they use."""

    def test_is_a_dict_and_serializes(self) -> None:
        headers = _model_with({"CF-RAY": "abc"}).headers
        assert isinstance(headers, dict)
        assert json.loads(json.dumps(headers)) == {"cf-ray": "abc"}
        assert dict(headers) == {"cf-ray": "abc"}

    @pytest.mark.parametrize("name", ["cf-ray", "CF-RAY", "Cf-Ray"])
    def test_lookup_in_any_casing(self, name: str) -> None:
        headers = _model_with({"CF-RAY": "abc"}).headers
        assert headers is not None
        assert headers[name] == "abc"
        assert headers.get(name) == "abc"
        assert name in headers

    def test_iteration_yields_one_casing(self) -> None:
        headers = _model_with({"X-RateLimit-Remaining": "5", "CF-RAY": "abc"}).headers
        assert headers is not None
        assert sorted(headers) == ["cf-ray", "x-ratelimit-remaining"]
        assert [k for k in headers if k.startswith("x-ratelimit")] == ["x-ratelimit-remaining"]

    def test_absent_header(self) -> None:
        headers = _model_with({"CF-RAY": "abc"}).headers
        assert headers is not None
        assert headers.get("missing") is None
        assert headers.get("MISSING", "fallback") == "fallback"
        assert "missing" not in headers
        with pytest.raises(KeyError):
            headers["MISSING"]

    def test_writes_are_case_insensitive(self) -> None:
        headers = _model_with({"CF-RAY": "abc"}).headers
        assert headers is not None
        headers["Cf-Ray"] = "new"
        assert len(headers) == 1
        assert headers["cf-ray"] == "new"
        assert headers.setdefault("CF-RAY", "ignored") == "new"
        assert headers.setdefault("X-New", "set") == "set"
        assert headers["x-new"] == "set"
        headers.update({"X-NEW": "updated"}, **{"X-Other": "other"})
        assert headers["x-new"] == "updated"
        assert headers["x-other"] == "other"
        assert headers.pop("X-OTHER") == "other"
        assert headers.pop("X-OTHER", "gone") == "gone"
        del headers["CF-RAY"]
        assert "cf-ray" not in headers

    def test_merging_stays_case_insensitive(self) -> None:
        headers = _model_with({"CF-RAY": "abc"}).headers
        assert headers is not None
        merged = headers | {"X-New": "v"}
        assert merged["X-NEW"] == "v"
        assert merged["CF-RAY"] == "abc"
        headers |= {"X-Other": "o"}
        assert headers["x-other"] == "o"
        assert "X-OTHER" in headers

    def test_copy_stays_case_insensitive(self) -> None:
        headers = _model_with({"CF-RAY": "abc"}).headers
        assert headers is not None
        assert headers.copy()["CF-RAY"] == "abc"

    def test_repeated_header_keeps_the_first_value(self) -> None:
        raw = CIMultiDict([("Set-Cookie", "a"), ("set-cookie", "b")])
        model = _Model()
        model._response = SimpleNamespace(headers=CIMultiDictProxy(raw))
        headers = model.headers
        assert headers is not None
        assert headers["set-cookie"] == dict(raw)["Set-Cookie"]

    def test_no_response_means_no_headers(self) -> None:
        assert _Model().headers is None
