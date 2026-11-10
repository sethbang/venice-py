"""Names the core packages export for annotating SDK return values."""

from venice_ai.core.models.base import CaseInsensitiveHeaders


def test_case_insensitive_headers_is_exported_beside_venice_base_model() -> None:
    import venice_ai.core
    import venice_ai.core.models

    for package in (venice_ai.core, venice_ai.core.models):
        assert "CaseInsensitiveHeaders" in package.__all__
        assert package.CaseInsensitiveHeaders is CaseInsensitiveHeaders
