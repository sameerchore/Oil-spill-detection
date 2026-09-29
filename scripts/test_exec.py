from app.api.routes import execute_pipeline
res = execute_pipeline("scene_mumbai_001")
print("Pipeline Status:", res.get("status"))
print("Slick Count:", len(res.get("detection", {}).get("features", [])))
print("Origin Time:", res.get("drift", {}).get("hindcast", {}).get("origin_time_iso"))
print("Age hours proxy:", res.get("drift", {}).get("hindcast", {}).get("age_hours_proxy"))
suspects = res.get("attribution", {}).get("suspects", [])
print("Suspects count:", len(suspects))
if suspects:
    for s in suspects[:5]:
        print(f"  Rank #{s['rank']}: {s['vessel_name']} ({s['mmsi']}) - Conf: {s['confidence_pct']:.1f}%, CPA: {s['distance_km']:.2f}km, Reasons: {s.get('reasons')}")