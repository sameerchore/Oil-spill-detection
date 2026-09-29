import sys
from app.metocean.store import load_metocean_data
from app.metocean.interpolator import MetoceanInterpolator
from app.drift.hindcast import run_hindcast
grid = load_metocean_data("scene_mumbai_001", center_lat=18.95, center_lon=72.15, t_sat="2026-09-02T12:00:00Z")
interp = MetoceanInterpolator(grid)
res = run_hindcast(interp, [(18.95, 72.15)], "2026-09-02T12:00:00Z", hindcast_hours=48, num_particles=50)
print("Origin time iso:", res["origin_time_iso"])
print("Origin hour found:", res["age_hours_proxy"])
for h in res["hourly_history"]:
    if h["hour_back"] in [1, 6, 12, 18, 21, 24, 36, 48]:
        print("Hour", h["hour_back"], "spread:", round(h["spread_km"], 2), "time:", h["time_iso"], "centroid:", round(h["centroid_lat"], 3), round(h["centroid_lon"], 3))