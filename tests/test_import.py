"""Test RAGLite."""

import weknow


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(weknow.__name__, str)
