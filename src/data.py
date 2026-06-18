from __future__ import annotations

from io import StringIO
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests


USGS_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"
REQUIRED_COLUMNS = ["time", "latitude", "longitude", "depth", "mag"]


def build_usgs_params(config: dict) -> dict:
    data_config = config["data"]
    return {
        "format": "csv",
        "eventtype": "earthquake",
        "starttime": data_config["start_time"],
        "endtime": data_config["end_time"],
        "minmagnitude": data_config["min_magnitude"],
        "minlatitude": data_config["min_latitude"],
        "maxlatitude": data_config["max_latitude"],
        "minlongitude": data_config["min_longitude"],
        "maxlongitude": data_config["max_longitude"],
        "orderby": "time-asc",
    }


def build_usgs_url(config: dict) -> str:
    params = build_usgs_params(config)
    return f"{USGS_ENDPOINT}?{urlencode(params)}"


def download_earthquake_data(config: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_params = build_usgs_params(config)
    limit = 20000
    offset = 1
    chunks: list[pd.DataFrame] = []

    while True:
        params = base_params | {"limit": limit, "offset": offset}
        response = requests.get(USGS_ENDPOINT, params=params, timeout=90)
        response.raise_for_status()
        chunk = pd.read_csv(StringIO(response.text))
        if chunk.empty:
            break

        chunks.append(chunk)
        if len(chunk) < limit:
            break
        offset += limit

    if not chunks:
        raise ValueError("No earthquake records were returned by the USGS API.")

    pd.concat(chunks, ignore_index=True).to_csv(output_path, index=False)
    return output_path


def load_earthquake_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for column in ["latitude", "longitude", "depth", "mag"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[df["type"].eq("earthquake")] if "type" in df.columns else df
    return df.sort_values("time").reset_index(drop=True)
