"""Phase 0 smoke test: the package installs and imports."""

import agentproof


def test_package_imports_and_has_version() -> None:
    assert agentproof.__version__ == "0.1.0"
