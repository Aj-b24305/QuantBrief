"""Pre-flight model availability check against a local Ollama registry.

The agent synthesizer queries ``GET {host}/api/tags`` *before* spending up to
30 seconds on a generation call, so a stopped Ollama daemon or an unpulled
model is detected immediately and routed to the cloud/deterministic fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

logger = logging.getLogger("quantbrief.llm.model_checker")

#: How long the availability probe itself may take (short by design — this is
#: just a registry listing, not a generation call).
DEFAULT_PROBE_TIMEOUT_SECONDS = 3.0


def ollama_native_tags_url(openai_base_url: str) -> str:
    """Map an OpenAI-compatible base URL to Ollama's native ``/api/tags`` URL.

    ``http://localhost:11434/v1`` -> ``http://localhost:11434/api/tags``.
    Any other base URL is treated as the server root and gets ``/api/tags``
    appended, which keeps the probe correct for raw ``http://host:11434``
    endpoints too.
    """
    url = (openai_base_url or "").strip().rstrip("/")
    url = url.removesuffix("/v1")
    return f"{url}/api/tags"


def _registered_model_matches(registered: str, requested: str) -> bool:
    """True when a registry entry satisfies the requested model name.

    Handles exact matches (``qwen3.5:9b`` == ``qwen3.5:9b``), prefixed
    equivalents (requested ``qwen3.5`` is served by ``qwen3.5:9b``), and
    optional registry namespaces (``library/qwen3.5:9b``).
    """
    req = requested.strip()
    reg = registered.strip()
    if "/" in reg:
        reg = reg.rsplit("/", 1)[-1]
    if not req or not reg:
        return False
    if reg == req:
        return True
    # "Prefixed equivalent": the request names a family, the registry a tag.
    return reg.startswith(f"{req}:") or (":" not in req and req.startswith(f"{reg.split(':')[0]}:"))


def _registry_contains(model_names: Sequence[str], model_name: str) -> bool:
    return any(_registered_model_matches(name, model_name) for name in model_names)


async def is_model_available(model_name: str, base_url: str, timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS) -> bool:
    """Check whether ``model_name`` is pullable from the local Ollama registry.

    Queries the native ``GET {base_url}/api/tags`` endpoint. Returns ``False``
    — never raises — when Ollama is unreachable, responds with a non-200
    status, or does not list the requested model (or a prefixed equivalent).
    """
    if not model_name or not model_name.strip():
        return False
    url = ollama_native_tags_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
    except Exception:  # noqa: BLE001 - unreachable daemon must not crash the caller
        logger.warning("Ollama pre-flight probe failed for %s at %s", model_name, url)
        return False

    if response.status_code != 200:
        logger.warning(
            "Ollama pre-flight probe for %s at %s returned HTTP %s",
            model_name,
            url,
            response.status_code,
        )
        return False

    try:
        payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        names = [str(item.get("name", "")) for item in models if isinstance(item, dict)]
    except Exception:  # noqa: BLE001 - unparseable body -> treat as unavailable
        logger.warning("Ollama pre-flight probe for %s returned an unparseable body", model_name)
        return False

    available = _registry_contains(names, model_name)
    if not available:
        logger.warning("Model %s is not in the local Ollama registry (%s)", model_name, url)
    return available
