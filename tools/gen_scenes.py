"""Generate full-bleed colour scenes, the archetype the build has been missing.

38% of sampled reference frames are full-bleed painted scenes rather than white
diagram cards, and the existing library backgrounds are all monochrome
leftovers from the superseded art direction (R=G=B, 90-98% paper). The palette
and finish below are measured from the reference: flat fills with a heavy
uniform black outline, no gradients, and a muted earthy range -- sage, ochre,
tan, warm brown, pale sky blue.

Scenes deliberately skip humanise(): that pass converts to greyscale, and its
tremor is tuned for thin ink line art, not flat colour fills.
"""
import os
from pathlib import Path

from why_engine.art.fal import LANDSCAPE, FalClient

SCENE_STYLE = (
    ", flat colour cartoon illustration, heavy uniform black outline, "
    "simple flat colour fills with no gradients and no shading, "
    "muted earthy palette of sage green, ochre, tan, warm brown and pale sky blue, "
    "full bleed edge to edge composition, hand drawn storybook look, no text, no words"
)

SCENES = {
    "scene_river_carp": "a wide midwestern american river at dawn seen from the bank, "
                        "thick with fish just under the surface, reeds and a low wooden jetty",
    "scene_depot_1877": "a small nineteenth century american railway depot platform, "
                        "wooden milk cans stacked on a handcart, a steam locomotive waiting",
    "scene_commission": "the interior of an 1880s american government office, "
                        "tall windows, wooden desks piled with paper, a wall map",
    "scene_fish_counter": "a modern american supermarket fish counter seen straight on, "
                          "ice trays of fish, price cards, fluorescent ceiling",
    "scene_choked_river": "a narrow muddy river completely choked with fish, "
                          "banks overgrown, a derelict fence, overcast sky",
}

def main():
    key = ""
    for line in Path(".env").read_text().splitlines():
        if line.startswith("FAL_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
    client = FalClient(key or os.environ.get("FAL_KEY", ""))
    out = Path("assets/scenes")
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, prompt) in enumerate(SCENES.items()):
        path = out / f"{name}.png"
        if path.exists():
            print(f"  skip {name} (exists)")
            continue
        path.write_bytes(client.generate(prompt + SCENE_STYLE, LANDSCAPE, seed=7100 + i))
        print(f"  wrote {path}")
    print(f"{client.image_count} images, ${client.spend_usd:.3f}")

if __name__ == "__main__":
    main()
