"""Resolve place names to latitude and longitude."""

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from strands import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


class PlaceGeocoder:
    """Client for resolving free-form place names to coordinates."""

    def get_coordinates(
        self,
        place_name: str,
        result_count: int = 1,
    ) -> list[dict[str, Any]]:
        """Return matching places with latitude and longitude."""
        normalized_place_name = place_name.strip()
        if not normalized_place_name:
            raise ValueError("place_name must not be empty")

        query_params = {
            "name": normalized_place_name,
            "count": max(1, min(result_count, 10)),
            "language": "en",
            "format": "json",
        }
        request = Request(
            f"{GEOCODING_URL}?{urlencode(query_params)}",
            headers={"Accept": "application/json"},
        )

        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            message = f"Open-Meteo geocoding request failed with status {exc.code}"
            raise RuntimeError(message) from exc
        except URLError as exc:
            message = f"Open-Meteo geocoding request failed: {exc.reason}"
            raise RuntimeError(message) from exc
        except json.JSONDecodeError as exc:
            message = "Open-Meteo geocoding response was not valid JSON"
            raise RuntimeError(message) from exc

        results = payload.get("results", []) if isinstance(payload, dict) else []
        if not isinstance(results, list):
            raise RuntimeError("Open-Meteo geocoding response had invalid results")

        places = []
        for result in results:
            if not isinstance(result, dict):
                continue

            latitude = result.get("latitude")
            longitude = result.get("longitude")
            if latitude is None or longitude is None:
                continue

            places.append(
                {
                    "name": result.get("name"),
                    "country": result.get("country"),
                    "admin1": result.get("admin1"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "timezone": result.get("timezone"),
                }
            )

        return places


@tool
def get_place_coordinates(
    place_name: str,
    result_count: int = 1,
) -> list[dict[str, Any]]:
    """Fetch latitude and longitude for a free-form place name."""
    return PlaceGeocoder().get_coordinates(place_name, result_count)
