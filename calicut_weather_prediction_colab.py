"""Google Colab-ready weather prediction workflow for Calicut/Kozhikode.

This script uses free Open-Meteo internet weather data and can be copied into a
single Colab cell. It intentionally avoids API keys so students can run it easily.
"""

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Location:
    """Latitude/longitude settings for the target city."""

    name: str = "Calicut / Kozhikode, Kerala"
    latitude: float = 11.2588
    longitude: float = 75.7804
    timezone: str = "Asia/Kolkata"


LOCATION = Location()
DAILY_TARGETS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
FEATURE_WINDOWS = [3, 7, 14, 30]
LAG_DAYS = [1, 2, 3, 7, 14]


def get_openmeteo_daily(url: str, params: dict) -> pd.DataFrame:
    """Download daily Open-Meteo data and convert the JSON response to a DataFrame."""
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    daily = response.json()["daily"]

    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for target in DAILY_TARGETS:
        df[target] = pd.to_numeric(daily[target], errors="coerce")
    return df


def fetch_historical_weather(years_back: int = 12) -> pd.DataFrame:
    """Fetch historical daily weather observations from the Open-Meteo Archive API."""
    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=365 * years_back)

    df = get_openmeteo_daily(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LOCATION.latitude,
            "longitude": LOCATION.longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": ",".join(DAILY_TARGETS),
            "timezone": LOCATION.timezone,
        },
    )
    return df.dropna().sort_values("date").reset_index(drop=True)


def fetch_future_dates(forecast_days: int = 16) -> pd.DataFrame:
    """Fetch future forecast dates from Open-Meteo so predictions align to calendar days."""
    df = get_openmeteo_daily(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LOCATION.latitude,
            "longitude": LOCATION.longitude,
            "forecast_days": forecast_days,
            "daily": ",".join(DAILY_TARGETS),
            "timezone": LOCATION.timezone,
        },
    )
    return df[["date"]]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, lag, and rolling weather features."""
    featured = df.copy().sort_values("date").reset_index(drop=True)
    featured["dayofyear"] = featured["date"].dt.dayofyear
    featured["month"] = featured["date"].dt.month
    featured["year"] = featured["date"].dt.year
    featured["sin_day"] = np.sin(2 * np.pi * featured["dayofyear"] / 365.25)
    featured["cos_day"] = np.cos(2 * np.pi * featured["dayofyear"] / 365.25)

    for target in DAILY_TARGETS:
        for lag in LAG_DAYS:
            featured[f"{target}_lag_{lag}"] = featured[target].shift(lag)
        for window in FEATURE_WINDOWS:
            shifted = featured[target].shift(1)
            featured[f"{target}_roll_mean_{window}"] = shifted.rolling(window).mean()
            featured[f"{target}_roll_std_{window}"] = shifted.rolling(window).std()

    return featured


def train_models(history: pd.DataFrame) -> Tuple[Dict[str, HistGradientBoostingRegressor], List[str], pd.DataFrame]:
    """Train one model per weather target and print holdout accuracy."""
    featured = add_features(history).dropna().reset_index(drop=True)
    feature_cols = [col for col in featured.columns if col not in ["date", *DAILY_TARGETS]]

    split = int(len(featured) * 0.85)
    train = featured.iloc[:split]
    test = featured.iloc[split:]

    models: Dict[str, HistGradientBoostingRegressor] = {}
    print(f"Location: {LOCATION.name}")
    print(f"Historical rows used: {len(history):,} ({history['date'].min().date()} to {history['date'].max().date()})")
    print("\nHoldout performance (lower is better):")

    for target in DAILY_TARGETS:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=500,
            l2_regularization=0.02,
            random_state=42,
        )
        model.fit(train[feature_cols], train[target])
        predictions = model.predict(test[feature_cols])
        mae = mean_absolute_error(test[target], predictions)
        rmse = float(np.sqrt(mean_squared_error(test[target], predictions)))
        unit = "mm" if target == "precipitation_sum" else "°C"
        print(f"- {target}: MAE={mae:.2f} {unit}, RMSE={rmse:.2f} {unit}")
        models[target] = model

    return models, feature_cols, featured


def predict_future(
    history: pd.DataFrame,
    future_dates: pd.DataFrame,
    models: Dict[str, HistGradientBoostingRegressor],
    feature_cols: List[str],
) -> pd.DataFrame:
    """Predict future daily weather recursively, using earlier predictions as lag inputs."""
    combined = history.copy().sort_values("date").reset_index(drop=True)
    predictions = []

    for future_date in pd.to_datetime(future_dates["date"]):
        row = {"date": future_date}
        for target in DAILY_TARGETS:
            row[target] = np.nan
        combined = pd.concat([combined, pd.DataFrame([row])], ignore_index=True)

        featured = add_features(combined)
        x_future = featured.iloc[[-1]][feature_cols]

        predicted_values = {"date": future_date}
        for target, model in models.items():
            value = float(model.predict(x_future)[0])
            if target == "precipitation_sum":
                value = max(0.0, value)
            predicted_values[target] = value
            combined.loc[combined.index[-1], target] = value

        predictions.append(predicted_values)

    return pd.DataFrame(predictions)


def plot_predictions(predictions: pd.DataFrame) -> None:
    """Plot predicted temperature and precipitation."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(predictions["date"], predictions["temperature_2m_max"], marker="o", label="Predicted max temp")
    axes[0].plot(predictions["date"], predictions["temperature_2m_min"], marker="o", label="Predicted min temp")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"Predicted daily weather for {LOCATION.name}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(predictions["date"], predictions["precipitation_sum"], label="Predicted rainfall")
    axes[1].set_ylabel("Rainfall (mm)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the complete Calicut weather prediction pipeline."""
    history = fetch_historical_weather(years_back=12)
    models, feature_cols, _ = train_models(history)
    future_dates = fetch_future_dates(forecast_days=16)
    predictions = predict_future(history, future_dates, models, feature_cols)

    print("\nPredictions for Calicut / Kozhikode:")
    print(predictions.assign(date=predictions["date"].dt.date).round(2).to_string(index=False))
    plot_predictions(predictions)


if __name__ == "__main__":
    main()
