"""Generate a topic's scenes and diagram icons in the live colour direction.

gen_scenes.py hardcoded one topic's five scenes. A channel making one video a
day cannot carry a tool edit per topic, so the set moves into art/<topic>.json
and this reads it.

Existing files are never overwritten: FLUX is stochastic, so regenerating an
asset that is already on disk produces a different-looking one and breaks the
continuity that makes the channel recognisable.
"""
import json
import os
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

from why_engine.art.fal import LANDSCAPE, PORTRAIT, FalClient
from why_engine.art.style import (
    COLOUR_STYLE, ICON_STYLE, icon_is_isolated, to_square_icon,
)


#: Rerolls allowed before an icon is kept as-is and reported.
ICON_TRIES = 4


def _key() -> str:
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FAL_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("FAL_KEY", "")


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: gen_assets.py art/<topic>.json", file=sys.stderr)
        return 2
    spec_path = Path(argv[0])
    spec = json.loads(spec_path.read_text())
    client = FalClient(_key())

    # Scenes are cycled by shot index with no regard to content, so each topic
    # keeps its own directory; icons are matched by name and can be shared.
    scenes_dir = Path("assets/scenes") / spec_path.stem
    icons_dir = Path("assets/icons")
    scenes_dir.mkdir(parents=True, exist_ok=True)
    icons_dir.mkdir(parents=True, exist_ok=True)

    for i, (name, prompt) in enumerate(sorted(spec.get("scenes", {}).items())):
        path = scenes_dir / f"{name}.png"
        if path.exists():
            print(f"  skip  {path}")
            continue
        path.write_bytes(client.generate(prompt + COLOUR_STYLE, LANDSCAPE, seed=8100 + i))
        print(f"  scene {path}")

    for i, (name, prompt) in enumerate(sorted(spec.get("icons", {}).items())):
        path = icons_dir / f"{name}.png"
        if path.exists():
            print(f"  skip  {path}")
            continue
        # Asking for a white background does not reliably get one, and an icon
        # with a room behind it pastes onto a white card as a rectangle of
        # someone else's picture. Reroll until it is actually cut out.
        icon, isolated = None, False
        for attempt in range(ICON_TRIES):
            raw = client.generate(prompt + ICON_STYLE, PORTRAIT,
                                  seed=8300 + i + attempt * 1000)
            icon = to_square_icon(Image.open(BytesIO(raw)))
            isolated = icon_is_isolated(icon)
            if isolated:
                break
        icon.save(path)
        print(f"  icon  {path}" + ("" if isolated else
                                   f"  NOT ISOLATED after {ICON_TRIES} tries"))

    print(f"{client.image_count} images, ${client.spend_usd:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
