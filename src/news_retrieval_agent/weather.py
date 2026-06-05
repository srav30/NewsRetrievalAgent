"""Retrieve weather forecasts from Open-Meteo."""

from typing import Any

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from strands import tool

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherForecast:
    """Client for fetching hourly weather forecast data from Open-Meteo."""

    def __init__(self, cache_path: str = ".cache", expire_after: int = 3600) -> None:
        cache_session = requests_cache.CachedSession(
            cache_path,
            expire_after=expire_after,
        )
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

    def get_hourly_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_hours: int = 5,
    ) -> list[dict[str, Any]]:
        """Return hourly temperature and precipitation forecast data."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ["temperature_2m", "precipitation_probability"],
        }

        try:
            responses = self.client.weather_api(OPEN_METEO_URL, params=params)
        except Exception as exc:
            raise RuntimeError("Open-Meteo request failed") from exc

        response = responses[0]
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation_probability = hourly.Variables(1).ValuesAsNumpy()

        hourly_times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        limit = max(1, min(forecast_hours, len(hourly_times)))
        forecast = []
        for index in range(limit):
            temperature = hourly_temperature_2m[index]
            precipitation = hourly_precipitation_probability[index]
            forecast.append(
                {
                    "time": hourly_times[index].isoformat(),
                    "temperature_2m": None
                    if pd.isna(temperature)
                    else round(float(temperature), 2),
                    "precipitation_probability": None
                    if pd.isna(precipitation)
                    else int(precipitation),
                }
            )

        return forecast


@tool
def get_weather_forecast(
    latitude: float,
    longitude: float,
    forecast_hours: int = 5,
) -> list[dict[str, Any]]:
    """Fetch hourly weather forecast data for latitude and longitude."""
    return WeatherForecast().get_hourly_forecast(latitude, longitude, forecast_hours)