from app.metocean.store import load_metocean_data
from app.metocean.interpolator import MetoceanInterpolator
from app.drift.hindcast import run_hindcast
from app.scoring.engine import AttributionScoringEngine
from app.api.routes import get_active_ais_store

grid = load_metocean_data('scene_mumbai_001', center_lat=18.95, center_lon=72.15, t_sat='2026-09-02T12:00:00Z')
interp = MetoceanInterpolator(grid)
res = run_hindcast(interp, [(18.95, 72.15)], '2026-09-02T12:00:00Z', hindcast_hours=48, num_particles=50, spread_threshold_km=3.0)

with get_active_ais_store() as store:
    engine = AttributionScoringEngine()
    suspects = engine.run_attribution_pipeline(
        store=store,
        origin_zone=res['origin_polygon_geojson'],
        origin_time=res['origin_time_iso'],
        radius_km=10.0,
        window_hours=3.0,
    )
    print('Keys:', list(suspects[0].keys()))
    for s in suspects:
        print('Rank #' + str(s['rank']) + ': ' + s['vessel_name'] + ' (' + str(s['mmsi']) + ') - Conf: ' + str(s.get('confidence_pct')) + '%, Flags: ' + str(s.get('anomaly_flags')))