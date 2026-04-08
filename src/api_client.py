"""Gamalytic API wrapper with rate limiting, retry, and logging.

Pro tier: unlimited daily requests. Only per-minute limits apply:
  - /game/{id}: 600 req/min
  - /steam-games/list (sort by id|revenue): 240 req/min
  - /steam-games/list (other sorts): 30 req/min
"""

import os
import time
import logging
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.gamalytic.com"

# Rate limits (requests per minute)
RATE_LIMITS = {
    "game_detail": 600,       # /game/{id}
    "list_sorted": 30,        # /steam-games/list with non-id/revenue sort
    "list_fast": 240,          # /steam-games/list sorted by id or revenue
}


class GamalyticClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GAMALYTIC_API_KEY")
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("Set GAMALYTIC_API_KEY in .env")
        self.session = requests.Session()
        self.session.params = {"api_key": self.api_key}
        self._request_count = 0
        self._error_count = 0
        self._last_request_time: dict[str, float] = {}

    def _rate_limit(self, endpoint_type: str):
        """Sleep if needed to respect per-minute rate limits."""
        rpm = RATE_LIMITS.get(endpoint_type, 30)
        min_interval = 60.0 / rpm
        last = self._last_request_time.get(endpoint_type, 0)
        elapsed = time.time() - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time[endpoint_type] = time.time()

    def _request(self, method: str, path: str, endpoint_type: str,
                 params: dict | None = None, max_retries: int = 5) -> Any:
        """Make an API request with rate limiting and retry."""
        url = f"{BASE_URL}{path}"
        for attempt in range(max_retries):
            self._rate_limit(endpoint_type)
            try:
                resp = self.session.request(method, url, params=params)
                self._request_count += 1

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 2 ** attempt * 5
                    logger.warning(
                        f"HTTP {resp.status_code} on {path}, "
                        f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})"
                    )
                    self._error_count += 1
                    time.sleep(wait)
                    continue

                if resp.status_code == 404:
                    logger.warning(f"404 Not Found: {path}")
                    return None

                resp.raise_for_status()

            except requests.exceptions.ConnectionError as e:
                wait = 2 ** attempt * 5
                logger.warning(f"Connection error on {path}: {e}, retrying in {wait}s")
                self._error_count += 1
                time.sleep(wait)

        logger.error(f"Max retries exceeded for {path}")
        self._error_count += 1
        return None

    def get_game_list(self, page: int = 0, limit: int = 1000,
                      sort: str = "id", sort_mode: str = "asc",
                      fields: str | None = None, **filters) -> dict | None:
        """Fetch a page of games from /steam-games/list."""
        endpoint_type = "list_fast" if sort in ("id", "revenue") else "list_sorted"
        params = {"page": page, "limit": limit, "sort": sort, "sort_mode": sort_mode}
        if fields:
            params["fields"] = fields
        params.update(filters)
        return self._request("GET", "/steam-games/list", endpoint_type, params=params)

    def get_game_details(self, steam_id: str, fields: str | None = None,
                         include_pre_release_history: bool = False) -> dict | None:
        """Fetch detailed game data from /game/{id}."""
        params = {}
        if fields:
            params["fields"] = fields
        if include_pre_release_history:
            params["include_pre_release_history"] = "true"
        return self._request("GET", f"/game/{steam_id}", "game_detail", params=params)

    @property
    def stats(self) -> str:
        return f"Requests: {self._request_count}, Errors: {self._error_count}"
