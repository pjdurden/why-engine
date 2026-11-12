from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # this repo supports 3.10, where tomllib isn't stdlib


def test_every_directly_imported_third_party_package_is_declared():
    """A clean install must be able to import why_engine.art."""
    data = tomllib.loads(Path("pyproject.toml").read_text())
    declared = " ".join(data["project"]["dependencies"]).lower()
    for pkg in ("pillow", "numpy", "httpx2", "pydantic", "anthropic"):
        assert pkg in declared, f"{pkg} is imported but not declared"
