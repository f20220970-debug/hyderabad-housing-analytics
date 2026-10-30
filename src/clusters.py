"""Locality clusters and growth gaps.

Each locality with enough listings is described by: median price per sq ft, number of listings (supply),
median size, share of 3BHK+, average amenity score and resale share. K-means on standardised features groups
them; k is chosen by silhouette score. Clusters are then named by their price and supply profile.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATS = ["median_psf", "listings", "median_area", "share_3bhk_plus", "amenity_score", "resale_share"]


def locality_table(df, min_listings=15):
    g = df.groupby("location")
    t = pd.DataFrame({"median_psf": g["price_per_sqft"].median(), "listings": g.size(),
                      "median_area": g["area"].median(), "share_3bhk_plus": g["bedrooms"].apply(lambda s: (s >= 3).mean()),
                      "amenity_score": g["amenity_score"].mean(), "resale_share": g["resale"].mean(),
                      "median_price_lakh": g["price"].median() / 1e5})
    return t[t["listings"] >= min_listings].copy()


def cluster(t, k_range=range(3, 7), seed=7):
    X = StandardScaler().fit_transform(np.column_stack([t["median_psf"], np.log(t["listings"]), t["median_area"],
                                                         t["share_3bhk_plus"], t["amenity_score"], t["resale_share"]]))
    best = max(k_range, key=lambda k: silhouette_score(X, KMeans(k, n_init=20, random_state=seed).fit_predict(X)))
    t = t.copy(); t["cluster"] = KMeans(best, n_init=20, random_state=seed).fit_predict(X)
    prof = t.groupby("cluster").agg(localities=("listings", "size"), listings=("listings", "sum"),
                                    median_psf=("median_psf", "median"), median_area=("median_area", "median"),
                                    share_3bhk_plus=("share_3bhk_plus", "mean"), amenity_score=("amenity_score", "mean"),
                                    resale_share=("resale_share", "mean"))
    city_psf, city_supply = t["median_psf"].median(), t["listings"].median()
    names = {}
    for c, r in prof.iterrows():
        price = "premium" if r.median_psf > 1.15 * city_psf else "affordable" if r.median_psf < 0.85 * city_psf else "mid-market"
        supply = "high-supply" if r.listings / r.localities > city_supply else "thin-supply"
        names[c] = f"{price}, {supply}"
    # if two clusters share a name, tell them apart by typical home size
    dupes = {n for n in names.values() if list(names.values()).count(n) > 1}
    city_area = t["median_area"].median()
    for c, r in prof.iterrows():
        if names[c] in dupes:
            names[c] += ", larger homes" if r.median_area > city_area else ", smaller homes"
    prof["name"] = prof.index.map(names); t["cluster_name"] = t["cluster"].map(names)
    return t, prof.reset_index(), best
