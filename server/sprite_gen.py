"""
Runtime sprite generation via Google Gemini ("Nano Banana").

Given the prompts Claude produced for a scene (one per character type, one per zone),
this generates matching 2D sprites on demand, cuts out their background to transparency,
and caches the PNGs on disk so repeat prompts are instant. The server streams each
finished PNG to Unity, which builds a Sprite at runtime.

Deps: google-genai, Pillow. Requires GEMINI_API_KEY in the environment.
Nothing here is imported at module load beyond the stdlib + Pillow; the Gemini client is
created lazily so the server still runs (in library-only mode) without a key installed.
"""

import os
import io
import hashlib
import logging
from pathlib import Path
from collections import deque

log = logging.getLogger("Director")

GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
# The public Gemini image model ("Nano Banana"). Override with GEMINI_IMAGE_MODEL if the
# model id changes (e.g. a newer Nano Banana revision).
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

CACHE_DIR = Path(__file__).resolve().parent / "sprite_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Shared style suffixes keep every sprite in a scene visually coherent. The semi-realistic
# 2.5D look lives in assetgen.style so the runtime path and the offline dataset pipeline
# stay identical; fall back to inline strings if the assetgen package isn't importable.
try:
    from assetgen.style import CHARACTER_SUFFIX as _CHAR_STYLE, SCENE_SUFFIX as _ZONE_STYLE
except Exception:
    _CHAR_STYLE = (
        ", full body, three-quarter top-down game angle, semi-realistic 2.5D game art style, "
        "soft studio lighting, centered, plain solid light grey background, no text."
    )
    _ZONE_STYLE = (
        ", 2.5D isometric-friendly top-down environment tile, semi-realistic game art, "
        "no people, no text, seamless-friendly edges."
    )

_client = None
_client_tried = False


def is_available() -> bool:
    """True if a Gemini key is present and the SDK imports — i.e. generate mode is usable."""
    return _get_client() is not None


def _get_client():
    global _client, _client_tried
    if _client_tried:
        return _client
    _client_tried = True
    if not GEMINI_API_KEY:
        log.warning("[sprite] GEMINI_API_KEY not set — runtime sprite generation disabled.")
        return None
    try:
        from google import genai  # noqa: import here so missing dep doesn't break the server
        _client = genai.Client(api_key=GEMINI_API_KEY)
        log.info(f"[sprite] Gemini client ready (model={GEMINI_IMAGE_MODEL}).")
    except Exception as e:
        log.error(f"[sprite] Could not init Gemini client: {e}")
        _client = None
    return _client


def _cache_path(kind: str, prompt: str) -> Path:
    h = hashlib.sha1(f"{kind}|{GEMINI_IMAGE_MODEL}|{prompt}".encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{kind}_{h}.png"


def _extract_image_bytes(response) -> bytes | None:
    try:
        for cand in getattr(response, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline else None
                if not data:
                    continue
                # SDK versions return either raw PNG bytes or a base64 str
                if isinstance(data, str):
                    import base64
                    try:
                        return base64.b64decode(data)
                    except Exception:
                        return data.encode("latin-1", "ignore")
                return data
    except Exception as e:
        log.error(f"[sprite] failed to read image from response: {e}")
    return None


def _call_gemini(client, styled: str):
    """Call the image model, trying an explicit IMAGE response modality first (required by
    some SDK/model versions), then a plain call. Returns the raw response or raises."""
    last_err = None
    try:
        from google.genai import types
        cfg = types.GenerateContentConfig(response_modalities=["IMAGE"])
        return client.models.generate_content(model=GEMINI_IMAGE_MODEL, contents=styled, config=cfg)
    except Exception as e:
        last_err = e
    try:
        return client.models.generate_content(model=GEMINI_IMAGE_MODEL, contents=styled)
    except Exception as e:
        raise last_err or e


def _cut_out_white(png_bytes: bytes, tol: int = 40) -> bytes:
    """Make the (edge-connected) white background transparent, keeping interior whites."""
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size
    px = img.load()

    def is_bg(c):
        return c[0] >= 255 - tol and c[1] >= 255 - tol and c[2] >= 255 - tol and c[3] > 0

    seen = bytearray(w * h)
    dq = deque()
    for x in range(w):
        dq.append((x, 0)); dq.append((x, h - 1))
    for y in range(h):
        dq.append((0, y)); dq.append((w - 1, y))

    while dq:
        x, y = dq.popleft()
        idx = y * w + x
        if seen[idx]:
            continue
        seen[idx] = 1
        c = px[x, y]
        if not is_bg(c):
            continue
        px[x, y] = (c[0], c[1], c[2], 0)
        if x > 0:     dq.append((x - 1, y))
        if x < w - 1: dq.append((x + 1, y))
        if y > 0:     dq.append((x, y - 1))
        if y < h - 1: dq.append((x, y + 1))

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _generate_sync(prompt: str, kind: str) -> bytes | None:
    """Blocking Gemini call + cutout. Call via asyncio.to_thread."""
    cache = _cache_path(kind, prompt)
    if cache.exists():
        try:
            return cache.read_bytes()
        except Exception:
            pass

    client = _get_client()
    if client is None:
        return None

    styled = prompt.strip() + (_CHAR_STYLE if kind == "character" else _ZONE_STYLE)
    try:
        response = _call_gemini(client, styled)
    except Exception as e:
        log.error(f"[sprite] Gemini generate failed ({kind}): {e}")
        return None

    raw = _extract_image_bytes(response)
    if not raw:
        log.error(f"[sprite] no image returned for {kind} prompt: {prompt[:60]!r}")
        return None

    if kind == "character":
        try:
            raw = _cut_out_white(raw)
        except Exception as e:
            log.warning(f"[sprite] background cutout failed, using opaque image: {e}")

    try:
        cache.write_bytes(raw)
    except Exception as e:
        log.warning(f"[sprite] cache write failed: {e}")
    return raw


async def generate_sprite(prompt: str, kind: str) -> bytes | None:
    """Async wrapper: returns transparent PNG bytes (character) or tile PNG bytes (zone)."""
    import asyncio
    if not prompt:
        return None
    return await asyncio.to_thread(_generate_sync, prompt, kind)


def render_styled(styled_prompt: str, cache_kind: str, cut_bg: bool) -> bytes | None:
    """Blocking render for the offline dataset pipeline (assetgen).

    The prompt is already fully styled by the caller, so no style suffix is appended here.
    Reuses the same disk cache and, when cut_bg is set, the white-background flood-fill.
    Returns PNG bytes or None (no key / no image / error). Safe to call from a thread pool.
    """
    if not styled_prompt:
        return None
    cache = _cache_path(cache_kind, styled_prompt)
    if cache.exists():
        try:
            return cache.read_bytes()
        except Exception:
            pass

    client = _get_client()
    if client is None:
        return None

    try:
        response = _call_gemini(client, styled_prompt)
    except Exception as e:
        log.error(f"[sprite] Gemini generate failed ({cache_kind}): {e}")
        return None

    raw = _extract_image_bytes(response)
    if not raw:
        log.error(f"[sprite] no image returned for {cache_kind} prompt: {styled_prompt[:60]!r}")
        return None

    if cut_bg:
        try:
            raw = _cut_out_white(raw)
        except Exception as e:
            log.warning(f"[sprite] background cutout failed, using opaque image: {e}")

    try:
        cache.write_bytes(raw)
    except Exception as e:
        log.warning(f"[sprite] cache write failed: {e}")
    return raw
