"""`youtube.search` tool (master spec section 19)."""

from __future__ import annotations

from typing import Any

from app.core.cache import TTLCache, make_cache_key
from app.core.errors import ValidationError
from app.integrations.youtube_api import YouTubeApiClient
from app.tools.base import Tool, ToolResult

_VALID_ORDERS = {"relevance", "date", "rating", "title", "viewCount"}


class SearchVideosTool(Tool):
    name = "youtube.search"
    description = "Search YouTube for videos matching a natural-language query. Returns structured video metadata."
    input_schema = {
        "query": {"type": "string", "required": True, "description": "What to search for."},
        "max_results": {"type": "integer", "required": False, "default": 10, "description": "1-25 results."},
        "order": {
            "type": "string",
            "required": False,
            "default": "relevance",
            "enum": sorted(_VALID_ORDERS),
        },
    }
    output_schema = {"results": {"type": "array", "items": "VideoSummary"}}
    permissions = []
    requires_confirmation = False

    def __init__(self, *, client: YouTubeApiClient, cache: TTLCache | None = None) -> None:
        self._client = client
        self._cache = cache

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        query = (arguments.get("query") or "").strip()
        if not query:
            raise ValidationError("`query` is required for youtube.search.")

        max_results = int(arguments.get("max_results", 10))
        order = arguments.get("order", "relevance")
        if order not in _VALID_ORDERS:
            raise ValidationError(f"`order` must be one of {sorted(_VALID_ORDERS)}.")

        access_token = arguments.get("access_token")
        cache_key = make_cache_key("youtube.search", query, max_results, order, bool(access_token))

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return ToolResult(success=True, data={"results": cached, "query": query, "cached": True})

        videos = self._client.search(query=query, max_results=max_results, order=order, access_token=access_token)
        results = [v.to_dict() for v in videos]

        if self._cache is not None:
            self._cache.set(cache_key, results)

        return ToolResult(success=True, data={"results": results, "query": query, "cached": False})
