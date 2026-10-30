"""Run everything on data/Hyderabad.csv (or --data PATH): price models, drivers, locality clusters, gaps."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import clusters as K  # noqa: E402
import load as L  # noqa: E402
import model as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(ROOT / "data/Hyderabad.csv")); a = ap.parse_args()
    df, amen = L.load(a.data)
    (ROOT / "results").mkdir(exist_ok=True); (ROOT / "charts").mkdir(exist_ok=True)
    cv = M.cross_validate(df, amen); drv, r2 = M.drivers(df)
    t, prof, k = K.cluster(K.locality_table(df))
    cv.round(3).to_csv(ROOT / "results/model_comparison.csv", index=False)
    drv.round(4).to_csv(ROOT / "results/price_drivers.csv")
    t.round(2).sort_values("median_psf", ascending=False).to_csv(ROOT / "results/localities.csv")
    prof.round(2).to_csv(ROOT / "results/cluster_profiles.csv", index=False)

    # gaps: premium-amenity localities with thin supply (inventory to win), and the largest high-supply hubs (lead volume)
    city_psf, city_sup = t["median_psf"].median(), t["listings"].median()
    gap_premium = t[(t["median_psf"] > city_psf) & (t["listings"] < city_sup)].sort_values("median_psf", ascending=False).head(5)
    hubs = t.sort_values("listings", ascending=False).head(5)
    resale = df.groupby("resale")["price_per_sqft"].median()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c, g in t.groupby("cluster_name"):
        ax.scatter(g["listings"], g["median_psf"], s=20 + 3 * g["share_3bhk_plus"] * 100, alpha=0.7, label=c)
    for loc, r in t.sort_values("listings", ascending=False).head(8).iterrows():
        ax.annotate(loc, (r["listings"], r["median_psf"]), fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("Listings (supply, log scale)"); ax.set_ylabel("Median price per sq ft (INR)")
    ax.set_title(f"Hyderabad localities: {k} clusters by price, supply, size and amenities"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "charts/locality_clusters.png", dpi=120); plt.close(fig)

    best = cv.sort_values("mae_lakh").iloc[0]; ols = cv.iloc[0]
    d = drv["coef"]
    lines = ["# Results summary", "",
             f"- Data: {len(df):,} Hyderabad listings after cleaning ({int(df['resale'].sum()):,} resale), {df['location'].nunique()} localities; "
             f"median price INR {df['price'].median() / 1e5:,.0f} lakh, median INR {df['price_per_sqft'].median():,.0f} per sq ft.",
             f"- Price prediction (5-fold CV): multivariate regression MAE INR {ols.mae_lakh:.1f} lakh, median error {100 * ols.median_ape:.0f}%, R2 {ols.r2_log:.2f}; "
             f"best model: {best.model} (MAE INR {best.mae_lakh:.1f} lakh, median error {100 * best.median_ape:.0f}%, R2 {best.r2_log:.2f}).",
             f"- Drivers (regression, R2 {r2:.2f}): 1% more area adds {d['log_area']:.2f}% to price; each extra bedroom {100 * (np.exp(d['bedrooms']) - 1):+.1f}%; "
             f"resale vs new {100 * (np.exp(d['resale']) - 1):+.1f}%; each extra listed amenity {100 * (np.exp(d['amenity_score']) - 1):+.1f}%; "
             f"locality price level is the strongest driver (coef {d['locality_level']:.2f}). Max VIF {drv['vif'].max():.1f}.",
             f"- Resale homes list at a median INR {resale.get(1, float('nan')):,.0f}/sq ft vs INR {resale.get(0, float('nan')):,.0f} for new.",
             f"- Locality clusters ({len(t)} localities with 15+ listings, k = {k}): " + "; ".join(
                 f"{r['name']} ({int(r['localities'])} localities, INR {r['median_psf']:,.0f}/sq ft)" for _, r in prof.iterrows()) + ".",
             "- Gap, premium but thin supply (inventory to win): " + ", ".join(f"{l} (INR {r.median_psf:,.0f}/sq ft, {int(r.listings)} listings)" for l, r in gap_premium.iterrows()) + ".",
             "- Volume hubs (lead volume, most competition): " + ", ".join(f"{l} ({int(r.listings)})" for l, r in hubs.iterrows()) + ".",
             ]
    (ROOT / "results/summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
