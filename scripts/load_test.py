from __future__ import annotations

import argparse
import asyncio
import collections
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

# Indian-city centroids (lat, lon) used to anchor synthetic coordinates
_CITIES = [
    (19.0760, 72.8777),  # Mumbai
    (18.5204, 73.8567),  # Pune
    (22.7196, 75.8577),  # Indore
    (28.6139, 77.2090),  # Delhi
    (12.9716, 77.5946),  # Bengaluru
]

_WEATHER = ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"]
_TRAFFIC = ["Low", "Medium", "High", "Jam"]
_ORDER_TYPE = ["Snack", "Meal", "Drinks", "Buffet"]
_VEHICLE_TYPE = ["motorcycle", "scooter", "electric_scooter", "bicycle"]
_CITY_TYPE = ["Urban", "Metropolitian", "Semi-Urban"]


def random_order(rng: random.Random) -> dict[str, Any]:
    # Delivery point is restaurant ± 0.03° so it's always nearby
    lat, lon = rng.choice(_CITIES)
    r_lat = lat + rng.uniform(-0.05, 0.05)
    r_lon = lon + rng.uniform(-0.05, 0.05)
    d_lat = r_lat + rng.uniform(-0.03, 0.03)
    d_lon = r_lon + rng.uniform(-0.03, 0.03)

    order_date = f"{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-2022"
    time_ordered = f"{rng.randint(8, 21):02d}:{rng.choice([0, 15, 30, 45]):02d}"

    return {
        "Delivery_person_Age": float(rng.randint(18, 40)),
        "Delivery_person_Ratings": round(rng.uniform(2.5, 5.0), 1),
        "Restaurant_latitude": r_lat,
        "Restaurant_longitude": r_lon,
        "Delivery_location_latitude": d_lat,
        "Delivery_location_longitude": d_lon,
        "Order_Date": order_date,
        "Time_Orderd": time_ordered,
        "Weather_conditions": rng.choice(_WEATHER),
        "Road_traffic_density": rng.choice(_TRAFFIC),
        "Vehicle_condition": rng.randint(0, 3),
        "Type_of_order": rng.choice(_ORDER_TYPE),
        "Type_of_vehicle": rng.choice(_VEHICLE_TYPE),
        "multiple_deliveries": float(rng.choices([0, 1, 2, 3], weights=[50, 30, 15, 5])[0]),
        "Festival": rng.choices(["No", "Yes"], weights=[90, 10])[0],
        "City": rng.choice(_CITY_TYPE),
    }


async def _ticker(rate_per_sec: float, duration_sec: float):
    # Anchored to wall-clock deadlines so RPS stays accurate under latency variance
    interval = 1.0 / rate_per_sec
    loop = asyncio.get_event_loop()
    deadline = loop.time()
    end = deadline + duration_sec
    while loop.time() < end:
        yield
        deadline += interval
        now = loop.time()
        if deadline > now:
            await asyncio.sleep(deadline - now)


async def health_check(client: httpx.AsyncClient, url: str) -> bool:
    try:
        r = await client.get(f"{url}/health", timeout=5.0)
        if r.status_code == 200:
            return True
        print(f"Health check returned {r.status_code} — aborting.", file=sys.stderr)
        return False
    except httpx.RequestError as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return False


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
    rng: random.Random,
    results: list[dict],
    timeout: float,
) -> None:
    async with sem:
        payload = random_order(rng)
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/predict", json=payload, timeout=timeout)
            results.append({"ok": r.is_success, "status": r.status_code, "latency_s": time.perf_counter() - t0})
        except httpx.RequestError:
            # status 0 = network/timeout error
            results.append({"ok": False, "status": 0, "latency_s": time.perf_counter() - t0})


async def run_load_test(args: argparse.Namespace) -> tuple[list[dict], float]:
    rng = random.Random(args.seed)
    sem = asyncio.Semaphore(20)  # cap in-flight requests
    results: list[dict] = []

    async with httpx.AsyncClient() as client:
        if not args.no_warmup:
            if not await health_check(client, args.url):
                sys.exit(1)

        rate_per_sec = args.rate / 60.0
        t_start = time.perf_counter()
        tasks = []
        async for _ in _ticker(rate_per_sec, args.duration):
            tasks.append(asyncio.create_task(send_request(client, args.url, sem, rng, results, args.timeout)))
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t_start

    return results, elapsed


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100)))
    return sorted_vals[k]


def print_summary(results: list[dict], elapsed: float, args: argparse.Namespace) -> None:
    successes = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    total = len(results)
    n_ok = len(successes)
    n_err = len(errors)

    latencies_ms = sorted(r["latency_s"] * 1000 for r in results)
    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    p99 = _percentile(latencies_ms, 99)
    lat_min = latencies_ms[0] if latencies_ms else 0.0
    lat_max = latencies_ms[-1] if latencies_ms else 0.0

    actual_rps = total / elapsed if elapsed > 0 else 0.0
    pct_ok = (n_ok / total * 100) if total else 0.0
    pct_err = (n_err / total * 100) if total else 0.0
    error_counts = collections.Counter(r["status"] for r in errors)

    sep = "=" * 60
    print(sep)
    print("  Food on the Fly — Load Test Results")
    print(sep)
    print(f"  Target rate   : {args.rate:.1f} req/min  ({args.rate / 60:.2f} req/s)")
    print(f"  Duration      : {args.duration:.1f} s")
    print(f"  URL           : {args.url}")
    print()
    print("  Requests")
    print(f"    Total        : {total:4d}")
    print(f"    Success (2xx): {n_ok:4d}  ({pct_ok:.1f}%)")
    print(f"    Errors        : {n_err:4d}  ({pct_err:.1f}%)")
    print(f"    Actual RPS    : {actual_rps:6.2f} req/s")
    print()
    print("  Latency (ms)")
    print(f"    p50          : {p50:7.1f}")
    print(f"    p95          : {p95:7.1f}")
    print(f"    p99          : {p99:7.1f}")
    print(f"    min          : {lat_min:7.1f}")
    print(f"    max          : {lat_max:7.1f}")
    if error_counts:
        print()
        print("  Errors by status")
        for status, count in sorted(error_counts.items()):
            label = str(status) if status != 0 else "network/timeout"
            print(f"    {label:<16} : {count:4d}")
    print(sep)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load test the Food on the Fly prediction API.")
    p.add_argument("--url", default="http://localhost:8080", help="API base URL")
    p.add_argument("--rate", type=float, default=100.0, help="Requests per minute")
    p.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")
    p.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout (seconds)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--no-warmup", action="store_true", help="Skip /health check")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results, elapsed = asyncio.run(run_load_test(args))
    print_summary(results, elapsed, args)


if __name__ == "__main__":
    main()
