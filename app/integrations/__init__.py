"""
Integration layer: external service clients (Gmail API, YouTube API,
Google OAuth, ...).

`google_oauth.py` (Phase 2) and `youtube_api.py` (Phase 3) are official-
API-only REST clients (master spec section 59: OFFICIAL API FIRST). The
pre-Phase-0 HTML-scraping fallback (`youtube_legacy.py`) has been removed;
`app.tools.youtube.search_videos.SearchVideosTool` is the only YouTube
lookup path now.
"""
