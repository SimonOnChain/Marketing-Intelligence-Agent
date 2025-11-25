"""Lightweight ETL helpers for preparing processed parquet artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config.settings import get_settings
from src.data.schema import TABLES


def load_raw_tables(data_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for name, table in TABLES.items():
        csv_path = data_dir / table.file
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}. Download the Olist dataset first.")
        df = pd.read_csv(csv_path, usecols=list(table.columns))
        frames[name] = df
    return frames


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(subset=["review_comment_message"]).copy()
    cleaned["review_comment_message"] = (
        cleaned["review_comment_message"].str.strip().str.replace(r"\s+", " ", regex=True)
    )
    cleaned["review_creation_date"] = pd.to_datetime(cleaned["review_creation_date"])
    cleaned["review_answer_timestamp"] = pd.to_datetime(cleaned["review_answer_timestamp"])
    return cleaned


def build_orders_view(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders_items = data["orders"].merge(data["order_items"], on="order_id", how="inner")
    orders_items = orders_items.merge(data["products"], on="product_id", how="left")
    orders_items = orders_items.merge(
        data["category_translation"], on="product_category_name", how="left"
    )
    orders_items = orders_items.merge(data["customers"], on="customer_id", how="left")
    orders_items["order_purchase_timestamp"] = pd.to_datetime(
        orders_items["order_purchase_timestamp"]
    )
    return orders_items


def save_parquet(df: pd.DataFrame, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.parquet"
    df.to_parquet(dest, index=False)
    return dest


def run_etl() -> dict[str, Path]:
    settings = get_settings()
    raw_dir = Path(settings.data_raw_dir)
    processed_dir = Path(settings.data_processed_dir)

    data = load_raw_tables(raw_dir)
    data["order_reviews"] = clean_reviews(data["order_reviews"])

    orders_view = build_orders_view(data)
    outputs = {
        "reviews": save_parquet(data["order_reviews"], "reviews", processed_dir),
        "orders_view": save_parquet(orders_view, "orders_view", processed_dir),
    }
    return outputs


if __name__ == "__main__":
    artifacts = run_etl()
    for name, path in artifacts.items():
        print(f"Saved {name}: {path}")

