"""Settings, and the two places a default can live.

from_env() carried its own literal fallbacks, so raising max_regenerations on
the model changed nothing: the orchestrator builds its Settings through
from_env, which quietly put the old value back. A default has to have one home.
"""
import pytest

from why_engine.config import Settings


def test_from_env_uses_the_model_default_when_unset(monkeypatch):
    monkeypatch.delenv("WE_MAX_REGEN", raising=False)
    assert Settings.from_env().max_regenerations == Settings().max_regenerations


def test_from_env_still_honours_the_environment(monkeypatch):
    monkeypatch.setenv("WE_MAX_REGEN", "9")
    assert Settings.from_env().max_regenerations == 9


def test_an_unparseable_override_is_not_silently_ignored(monkeypatch):
    """Falling back to the default on a typo would run a daily pipeline on
    settings nobody chose."""
    monkeypatch.setenv("WE_MAX_REGEN", "lots")
    with pytest.raises(ValueError):
        Settings.from_env()
