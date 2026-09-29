from app.api.routes import _process_detection, load_metocean_data, get_active_ais_store
from app.metocean.interpolator import MetoceanInterpolator
from app.drift.hindcast import run_hindcast
from app.scoring.engine import AttributionScoringEngine

detection_fc = _process_detection(scene_id='scene_mumbai_001')
slicks = [f for f in detection_fc['features'] if f['properties']['class_id'] == 2]
slick_geom = slicks[0]['geometry']

grid = load_metocean_data('scene_mumbai_001', center_lat=18.95, center_lon=72.15, t_sat='2026-09-02T12:00:00Z')
interp = MetoceanInterpolator(grid)
res = run_hindcast(interp, slick_geom, '2026-09-02T12:00:00Z', hindcast_hours=48, num_particles=50, spread_threshold_km=4.42)

print('Origin time:', res['origin_time_iso'])
print('Age hours proxy:', res['age_hours_proxy'])

with get_active_ais_store() as store:
    engine = AttributionScoringEngine()
    suspects = engine.run_attribution_pipeline(
        store=store,
        origin_zone=res['origin_polygon_geojson'],
        origin_time=res['origin_time_iso'],
        radius_km=10.0,
        window_hours=3.0,
    )
    print('Suspects count:', len(suspects))
    for s in suspects:
        print('Rank #' + str(s['rank']) + ': ' + s['vessel_name'] + ' (MMSI ' + str(s['mmsi']) + ') - Conf: ' + str(round(s['confidence_pct'], 1)) + '%, S_total: ' + str(round(s['s_total'], 3)) + ', Flags: ' + str(s['anomaly_flags']) + ', Dist: ' + str(round(s['distance_km'], 2)) + ' km')