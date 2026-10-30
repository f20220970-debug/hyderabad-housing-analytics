"""FAKE data in the Kaggle Hyderabad.csv layout, to test the pipeline (and run CI) without the real file."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
AMEN = ["MaintenanceStaff", "Gymnasium", "SwimmingPool", "LandscapedGardens", "JoggingTrack", "RainWaterHarvesting",
        "IndoorGames", "ShoppingMall", "Intercom", "SportsFacility", "ATM", "ClubHouse", "School", "24X7Security",
        "PowerBackup", "CarParking", "StaffQuarter", "Cafeteria", "MultipurposeRoom", "Hospital", "WashingMachine",
        "Gasconnection", "AC", "Wifi", "Children'splayarea", "LiftAvailable", "BED", "VaastuCompliant", "Microwave",
        "GolfCourse", "TV", "DiningTable", "Sofa", "Wardrobe", "Refrigerator"]
locs = {f"Locality {i}": rng.uniform(3500, 11000) for i in range(40)}
rows = []
for _ in range(2500):
    loc = rng.choice(list(locs)); bed = int(rng.choice([1, 2, 3, 4], p=[.1, .45, .38, .07]))
    area = float(rng.normal(500 + 450 * bed, 150)); resale = int(rng.random() < 0.3)
    known = rng.random() < 0.6
    am = rng.integers(0, 2, len(AMEN)) if known else np.full(len(AMEN), 9)
    psf = locs[loc] * (1 + 0.004 * (am.sum() if known else 0)) * (0.93 if resale else 1) * rng.lognormal(0, 0.12)
    rows.append([round(psf * max(area, 300)), round(max(area, 300)), loc, bed, resale, *am])
pd.DataFrame(rows, columns=["Price", "Area", "Location", "No. of Bedrooms", "Resale", *AMEN]).to_csv(
    Path(__file__).resolve().parents[1] / "data/Hyderabad.csv", index=False)
print("fake data written")
