import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clusters as K  # noqa: E402
import load as L  # noqa: E402
import model as M  # noqa: E402


def _csv(tmp_path):
    rng = np.random.default_rng(0); n = 600
    df = pd.DataFrame({"Price": rng.integers(3_000_000, 15_000_000, n), "Area": rng.integers(600, 2500, n),
                       "Location": rng.choice([f"L{i}" for i in range(20)], n), "No. of Bedrooms": rng.integers(1, 5, n),
                       "Resale": rng.integers(0, 2, n), "Gymnasium": rng.choice([0, 1, 9], n), "Lift Available": rng.choice([0, 1, 9], n)})
    p = tmp_path / "h.csv"; df.to_csv(p, index=False); return p


def test_load_normalises_names_and_codes_9_as_missing(tmp_path):
    df, amen = L.load(_csv(tmp_path))
    assert {"price", "area", "location", "bedrooms", "resale"} <= set(df.columns)
    assert "liftavailable" in amen and (df[amen] == 9).sum().sum() == 0


def test_locality_encoding_uses_train_only():
    tr = pd.DataFrame({"location": ["A", "A", "B"], "price_per_sqft": [1000, 1000, 4000]})
    te = pd.DataFrame({"location": ["A", "C"], "price_per_sqft": [9999, 9999]})
    _, enc_te = M.locality_level(tr, te)
    assert enc_te[1] == pytest.approx(np.log([1000, 1000, 4000]).mean())    # unseen locality -> city mean
    assert enc_te[0] < np.log(4000)                                           # test prices never leak in


def test_clusters_and_cv_run(tmp_path):
    df, amen = L.load(_csv(tmp_path))
    t, prof, k = K.cluster(K.locality_table(df, min_listings=10), k_range=range(3, 5))
    assert prof["localities"].sum() == len(t) and 3 <= k <= 4
    cv = M.cross_validate(df, amen, folds=3)
    assert set(cv["model"]) == {"Multivariate regression (OLS)", "Gradient boosting"}
