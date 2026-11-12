import os

from pydantic import BaseModel

#: Environment overrides, by field name. Values fall back to the field default
#: rather than to a second literal: from_env() used to carry its own copy, so
#: raising a default on the model silently did nothing.
ENV_OVERRIDES = {"max_regenerations": "WE_MAX_REGEN"}


class Settings(BaseModel):
    model: str = "claude-opus-5"
    # Six, not three. A script carries eighty-odd claims, and a rewrite that
    # fixes the one thing it was told about tends to break something small
    # elsewhere; three attempts ran out while still converging.
    max_regenerations: int = 6
    min_runtime_s: int = 600
    max_runtime_s: int = 1320
    max_act_s: int = 300
    words_per_second: float = 2.6

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(**{
            field: os.environ[var]
            for field, var in ENV_OVERRIDES.items()
            if var in os.environ
        })
