"""Every resource reachable on the client must also be importable by name.

``client.decisions`` worked while ``from venice_ai.resources import Decisions``
raised ``ImportError`` — the resource module existed and was wired into
:class:`~venice_ai.VeniceClient`, but nobody had added it to the package's
import block. Nothing in the test suite noticed, because every test reached the
resource through the client.

These tests derive their expectations from the client's own annotations rather
than from a hand-kept list, so a new resource is covered the moment it is
wired up.
"""

import venice_ai
import venice_ai.resources as resources_pkg
from venice_ai import VeniceClient

# Resources whose class name is not just the attribute name title-cased.
_ATTR_TO_CLASS = {"chat": "ChatCompletions", "api_keys": "ApiKeys", "x402": "X402"}


def _resource_attrs() -> dict[str, str]:
    """Map client attribute name -> resource class name, from the annotations."""
    hints = VeniceClient.__annotations__
    out = {}
    for attr, annotation in hints.items():
        if attr.startswith("_") or attr == "rate_limiter":
            continue
        name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
        # `chat` is annotated ChatResource, a façade over ChatCompletions.
        out[attr] = _ATTR_TO_CLASS.get(attr, name)
    return out


class TestResourceExportParity:
    def test_every_client_resource_is_importable_from_the_package(self):
        missing = [
            f"{attr} -> {cls}"
            for attr, cls in _resource_attrs().items()
            if cls not in resources_pkg.__all__ or not hasattr(resources_pkg, cls)
        ]
        assert not missing, f"wired onto the client but not exported: {missing}"

    def test_resources_dunder_all_has_no_dangling_names(self):
        assert [n for n in resources_pkg.__all__ if not hasattr(resources_pkg, n)] == []

    def test_top_level_resource_exports_agree_with_the_package(self):
        """Resources reached directly from ``venice_ai`` (only a few are) must
        not be names the resources package itself has forgotten to export."""
        top_level_resources = {
            n for n in venice_ai.__all__ if n in dir(resources_pkg) and n[0].isupper()
        }
        assert top_level_resources <= set(resources_pkg.__all__), (
            f"exported from venice_ai but missing from venice_ai.resources.__all__: "
            f"{sorted(top_level_resources - set(resources_pkg.__all__))}"
        )
