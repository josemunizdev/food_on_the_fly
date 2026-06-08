"""Concept drift simulation"""

import argparse
import json
import statistics
import urllib.request

BASE = {
    "Delivery_person_Age": 30.0,
    "Delivery_person_Ratings": 4.5,
    "Restaurant_latitude": 22.745049,
    "Restaurant_longitude": 75.892471,
    "Delivery_location_latitude": 22.765049,
    "Delivery_location_longitude": 75.912471,
    "Order_Date": "19-03-2022",
    "Time_Orderd": "12:00",
    "Weather_conditions": "Sunny",
    "Road_traffic_density": "Low",
    "Vehicle_condition": 2,
    "Type_of_order": "Snack",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 0.0,
    "Festival": "No",
    "City": "Urban",
}

# Adversarial: worst-case conditions to simulate a drifted distribution
DRIFT = {
    **BASE,
    "Delivery_person_Age": 20.0,
    "Delivery_person_Ratings": 2.0,
    "Delivery_location_latitude": 22.945049,
    "Delivery_location_longitude": 76.092471,
    "Time_Orderd": "19:00",
    "Weather_conditions": "Stormy",
    "Road_traffic_density": "Jam",
    "Vehicle_condition": 0,
    "multiple_deliveries": 3.0,
    "Festival": "Yes",
}


def batch_predict(url: str, record: dict, n: int) -> list[float]:
    payload = json.dumps({"instances": [record] * n}).encode()
    req = urllib.request.Request(
        f"{url}/predict/batch",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["predictions"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    print(f"Target: {args.url}\n")

    baseline_preds = batch_predict(args.url, BASE, args.n)
    drift_preds = batch_predict(args.url, DRIFT, args.n)

    b_mean = statistics.mean(baseline_preds)
    d_mean = statistics.mean(drift_preds)
    spike_pct = (d_mean - b_mean) / b_mean * 100

    print(f"Baseline  — mean: {b_mean:.1f} min  (n={len(baseline_preds)})")
    print(f"Drift     — mean: {d_mean:.1f} min  (n={len(drift_preds)})")
    print(f"Spike:  +{spike_pct:.1f}%")

    if spike_pct > 20:
        print("\n[ALERT] Prediction drift detected — mean shifted > 20%")


if __name__ == "__main__":
    main()
