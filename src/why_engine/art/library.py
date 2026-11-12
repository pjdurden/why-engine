"""The one-time reusable asset library.

Generated once and amortised across every video. Recurring elements are the
same PNG in every shot, which is what makes the embedding-distance check from
the original design unnecessary.
"""
import argparse
import io
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from why_engine.art.fal import LANDSCAPE, PORTRAIT, FalClient
from why_engine.art.humanise import humanise, is_monochrome

STYLE_SUFFIX = (
    ", simple hand drawn doodle, thin black ink line on plain white paper, "
    "flat straight-on view, no colour, no shading, no crosshatching, no background detail"
)


class Asset(BaseModel):
    name: str = Field(min_length=1)
    kind: Literal["cast", "background", "prop"]
    prompt: str = Field(min_length=1)
    size: str


def _cast(name: str, prompt: str) -> Asset:
    return Asset(name=name, kind="cast", prompt=prompt, size=PORTRAIT)


EVERYMAN = (
    "an ordinary adult man, medium large round head, flat hair with one cowlick, "
    "small dot eyes, tiny line nose, plain collared shirt and trousers, "
    "thin noodle arms and legs, full body"
)
BUREAUCRAT = (
    "a very tall thin bureaucrat in a plain suit and tie, small head, blank face "
    "with no visible eyes, stiff formal posture, thin noodle limbs, full body"
)

CAST: List[Asset] = [
    _cast("everyman_neutral", EVERYMAN + ", standing still, neutral expression"),
    _cast("everyman_confused", EVERYMAN + ", shrugging, confused expression"),
    _cast("everyman_pointing", EVERYMAN + ", pointing to one side"),
    _cast("everyman_surprised", EVERYMAN + ", arms raised, surprised expression"),
    _cast("everyman_thinking", EVERYMAN + ", hand on chin, thinking"),
    _cast("everyman_disgusted", EVERYMAN + ", recoiling, disgusted expression"),
    _cast("bureaucrat_neutral", BUREAUCRAT + ", standing still"),
    _cast("bureaucrat_pointing", BUREAUCRAT + ", pointing sternly"),
    _cast("bureaucrat_clipboard", BUREAUCRAT + ", holding a clipboard"),
    _cast("bureaucrat_stamping", BUREAUCRAT + ", stamping a document"),
]

BACKGROUNDS: List[Asset] = [
    Asset(name="bg_" + n, kind="background", prompt=p, size=LANDSCAPE)
    for n, p in [
        ("empty", "an empty plain room, minimal outline"),
        ("river", "a simple river bank with a few reeds"),
        ("street", "a plain suburban street with two simple houses"),
        ("office", "a plain government office with a desk and a filing cabinet"),
        ("field", "an open farm field with a simple fence"),
        ("factory", "a plain factory interior with simple machines"),
    ]
]

PROPS: List[Asset] = [
    Asset(name="prop_" + n, kind="prop", prompt=p, size=PORTRAIT)
    for n, p in [
        ("document", "a single sheet of paper with a few lines of writing"),
        ("chart", "a simple line chart going up"),
        ("money", "a plain banknote"),
        ("crate", "a plain wooden crate"),
    ]
]

ALL_ASSETS: List[Asset] = CAST + BACKGROUNDS + PROPS


def build_prompt(asset: Asset) -> str:
    return asset.prompt + STYLE_SUFFIX


def _is_readable_png(path: Path) -> bool:
    """A file that exists but cannot be decoded is worse than one that is absent:
    the build skips it forever and the verifier used to crash on it."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def build_library(assets: Sequence[Asset], client: FalClient, out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for index, asset in enumerate(assets):
        path = out_dir / f"{asset.name}.png"
        if not _is_readable_png(path):  # absent OR corrupt: regenerate
            raw = client.generate(build_prompt(asset), image_size=asset.size)
            humanise(Image.open(io.BytesIO(raw)), seed=index).save(path)
        paths.append(path)
    return paths


def verify_library(out_dir: Path) -> List[str]:
    """Every asset must be pure black on white. Anything else breaks the contrast
    with archival photographs, which are the only thing allowed to carry tone."""
    problems: List[str] = []
    for path in sorted(Path(out_dir).glob("*.png")):
        if not _is_readable_png(path):
            problems.append(f"{path.name}: unreadable or corrupt")
            continue
        img = Image.open(path)
        ink = (np.array(img.convert("L")) < 200).mean()
        if ink > 0.98 or ink < 0.002:
            problems.append(f"{path.name}: blank or solid ({ink:.0%} ink)")
            continue
        if not is_monochrome(img):
            problems.append(f"{path.name}: not monochrome")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="why-engine-library")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", action="append", default=None,
                        help="build only the named asset; repeatable")
    args = parser.parse_args(argv)

    key = os.environ.get("FAL_KEY", "")
    if not key:
        print("FAL_KEY is not set; put it in ~/why-engine/.env", file=sys.stderr)
        return 1

    assets = ALL_ASSETS
    if args.only:
        wanted = set(args.only)
        known = {a.name for a in ALL_ASSETS}
        unknown = sorted(wanted - known)
        if unknown:
            print(f"unknown asset(s): {', '.join(unknown)}", file=sys.stderr)
            return 1
        assets = [a for a in ALL_ASSETS if a.name in wanted]

    client = FalClient(key)
    build_library(assets, client, args.out)
    print(f"built {len(assets)} asset(s); spend ${client.spend_usd:.3f}")

    problems = verify_library(args.out)
    if problems:
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
