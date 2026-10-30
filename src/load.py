"""Load and clean the Hyderabad listings (Kaggle: "Housing Prices in Metropolitan Areas of India", Hyderabad.csv).

The scraped file codes amenities as 1 (present), 0 (absent) and 9 (not mentioned). Column names are
normalised so small naming differences between versions don't break the pipeline.
"""
import re

import numpy as np
import pandas as pd

CORE = {"price": "price", "area": "area", "location": "location", "noofbedrooms": "bedrooms", "resale": "resale"}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", c.lower())


def load(path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rename = {c: CORE.get(_norm(c), _norm(c)) for c in raw.columns}
    df = raw.rename(columns=rename)
    missing = [c for c in CORE.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found: {missing}. Found: {list(df.columns)[:12]}...")
    amen = [c for c in df.columns if c not in CORE.values()]
    df[amen] = df[amen].replace(9, np.nan)
    df["location"] = df["location"].astype(str).str.strip().str.title()
    df = df[(df["price"] > 0) & (df["area"] > 100) & (df["bedrooms"].between(1, 6))].copy()
    df["price_per_sqft"] = df["price"] / df["area"]
    lo, hi = df["price_per_sqft"].quantile([0.01, 0.99])          # drop obvious data-entry errors
    df = df[df["price_per_sqft"].between(lo, hi)].copy()
    df["amenities_known"] = df[amen].notna().any(axis=1)
    df["amenity_score"] = df[amen].fillna(0).sum(axis=1)
    return df.reset_index(drop=True), amen
