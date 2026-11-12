"""fal.ai FLUX.1 schnell client.

fal bills per megapixel rounded UP, so only sub-megapixel presets are used:
portrait_16_9 (576x1024) and landscape_16_9 (1024x576) both bill at the 1 MP
tier. square_hd is 1.05 MP and would bill double for less usable area.
"""
from typing import Any, Callable, Dict, Optional

ENDPOINT = "https://fal.run/fal-ai/flux/schnell"
PORTRAIT = "portrait_16_9"
LANDSCAPE = "landscape_16_9"
PERMITTED_SIZES = frozenset({PORTRAIT, LANDSCAPE})
USD_PER_IMAGE = 0.003
TIMEOUT_S = 120


def _http(*, url: str, headers: Dict[str, str], payload=None, download: bool = False):
    import httpx2 as httpx

    if download:
        r = httpx.get(url, timeout=TIMEOUT_S, follow_redirects=True)
        r.raise_for_status()
        return r.content
    r = httpx.post(url, headers=headers, json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


class FalClient:
    def __init__(self, api_key: str, transport: Optional[Callable[..., Any]] = None):
        if not api_key:
            raise ValueError("FAL_KEY is empty; set it in ~/why-engine/.env")
        self._key = api_key
        self._transport = transport or _http
        self.image_count = 0
        self.spend_usd = 0.0

    def generate(self, prompt: str, image_size: str = PORTRAIT, seed: Optional[int] = None) -> bytes:
        if image_size not in PERMITTED_SIZES:
            raise ValueError(
                f"image_size {image_size!r} is not permitted: USD_PER_IMAGE assumes a "
                f"sub-megapixel preset, and fal bills per megapixel rounded up. "
                f"Permitted: {sorted(PERMITTED_SIZES)}"
            )
        payload: Dict[str, Any] = {"prompt": prompt, "image_size": image_size, "num_images": 1}
        if seed is not None:
            payload["seed"] = seed
        body = self._transport(
            url=ENDPOINT,
            headers={"Authorization": f"Key {self._key}", "Content-Type": "application/json"},
            payload=payload,
        )
        images = body.get("images") if isinstance(body, dict) else None
        url = None
        if isinstance(images, list) and images and isinstance(images[0], dict):
            url = images[0].get("url")
        if not url:
            raise RuntimeError(f"fal returned no image: {str(body)[:200]}")
        data = self._transport(url=url, headers={}, download=True)
        self.image_count += 1
        self.spend_usd += USD_PER_IMAGE
        return data
