from __future__ import annotations

import http.client
import json
import logging
import os
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from loaders.base import DataLoader

logger = logging.getLogger(__name__)

DEFAULT_RAPIDAPI_HOST = "twitter-api45.p.rapidapi.com"
DEFAULT_QUERY = "bitcoin prediction"
DEFAULT_SEARCH_TYPE = "Top"
DEFAULT_ENDPOINT = "/search.php"
DEFAULT_TWEET_COUNT = 300
MAX_REQUEST_ROUNDS = 50


class RapidAPITwitterLoader(DataLoader):
    def __init__(
        self,
        timeout: int = 20,
        tweet_count: int = DEFAULT_TWEET_COUNT,
    ):
        if tweet_count < 1:
            raise ValueError("tweet_count debe ser >= 1.")
        self.host = os.getenv("RAPIDAPI_HOST") or DEFAULT_RAPIDAPI_HOST
        self.timeout = timeout
        self.tweet_count = tweet_count

    def load(self) -> pd.DataFrame:
        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            raise ValueError("Falta RAPIDAPI_KEY en el entorno (.env).")

        logger.info(
            "RapidAPI Twitter: inicio carga (host=%s, query=%r, objetivo=%s tweets)",
            self.host,
            DEFAULT_QUERY,
            self.tweet_count,
        )

        raw_tweets: list[dict[str, Any]] = []
        cursor: str | None = None
        rounds = 0

        while len(raw_tweets) < self.tweet_count and rounds < MAX_REQUEST_ROUNDS:
            rounds += 1
            params: dict[str, str] = {
                "query": DEFAULT_QUERY,
                "search_type": DEFAULT_SEARCH_TYPE,
            }

            if cursor:
                params["cursor"] = cursor

            logger.debug(
                "RapidAPI Twitter: petición %s/%s (tweets acumulados=%s, cursor=%s)",
                rounds,
                MAX_REQUEST_ROUNDS,
                len(raw_tweets),
                "sí" if cursor else "no (primera página)",
            )

            payload = self._fetch_json(api_key, params)
            batch = self._timeline_tweets(payload)
            raw_tweets.extend(batch)

            logger.info(
                "RapidAPI Twitter: página %s --> +%s tweets (total=%s)",
                rounds,
                len(batch),
                len(raw_tweets),
            )

            if len(raw_tweets) >= self.tweet_count:
                logger.info(
                    "RapidAPI Twitter: objetivo alcanzado (%s tweets).",
                    self.tweet_count,
                )
                break
            if not batch:
                logger.info("RapidAPI Twitter: timeline vacío, fin de paginación.")
                break

            cursor = (payload.get("next_cursor") or "").strip() or None
            if not cursor:
                logger.info(
                    "RapidAPI Twitter: sin next_cursor (total=%s tweets).",
                    len(raw_tweets),
                )
                break

        if rounds >= MAX_REQUEST_ROUNDS and len(raw_tweets) < self.tweet_count:
            logger.warning(
                "RapidAPI Twitter: límite de peticiones (%s) sin llegar a %s "
                "tweets (total=%s).",
                MAX_REQUEST_ROUNDS,
                self.tweet_count,
                len(raw_tweets),
            )

        trimmed = raw_tweets[: self.tweet_count]
        rows = [self._normalize_tweet(t) for t in trimmed]
        logger.info("RapidAPI Twitter: DataFrame con %s filas.", len(rows))
        return pd.DataFrame(rows)

    def _fetch_json(self, api_key: str, query_params: dict[str, str]) -> dict[str, Any]:
        query_string = urlencode(query_params, doseq=True)
        path = f"{DEFAULT_ENDPOINT}?{query_string}"
        headers = {
            "x-rapidapi-host": self.host,
            "x-rapidapi-key": api_key,
            "Content-Type": "application/json",
        }

        conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        raw_data = response.read()
        conn.close()

        if response.status >= 400:
            body = raw_data.decode("utf-8", errors="replace")
            raise RuntimeError(f"RapidAPI error {response.status}: {body}")

        data = json.loads(raw_data.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("La API debe devolver un JSON objeto con timeline.")
        return data

    def _timeline_tweets(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        timeline = payload.get("timeline") or []
        if not isinstance(timeline, list):
            return []
        return [x for x in timeline if isinstance(x, dict)]

    def _normalize_tweet(self, tweet: dict[str, Any]) -> dict[str, Any]:
        return {
            "username": tweet.get("screen_name") or "",
            "text": tweet.get("text") or "",
            "date": tweet.get("created_at"),
        }
