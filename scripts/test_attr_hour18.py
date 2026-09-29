from app.metocean.store import load_metocean_data
from app.metocean.interpolator import MetoceanInterpolator
from app.drift.hindcast import run_hindcast
from app.scoring.engine import AttributionScoringEngine
from app.api.routes import get_active_ais_store

grid = load_metocean_data('scene_mumbai_001', center_lat=18.95, center_lon=72.15, t_sat='2026-09-02T12:00:00Z')
interp = MetoceanInterpolator(grid)
res = run_hindcast(interp, [(18.95, 72.15)], '2026-09-02T12:00:00Z', hindcast_hours=18, num_particles=50)

with get_active_ais_store() as store:
    engine = AttributionScoringEngine()
    suspects = engine.run_attribution_pipeline(
        store=store,
        origin_zone=res['origin_polygon_geojson'],
        origin_time='2026-09-01T18:00:00Z',
        radius_km=10.0,
        window_hours=3.0,
    )
    print('Suspects at 18h:', len(suspects))
    for s in suspects[:5]:
        print('Rank #' + str(s['rank']) + ': ' + s['vessel_name'] + ' (' + str(s['mmsi']) + ') - Conf: ' + str(round(s['confidence_pct'], 1)) + '%, Flags: ' + str(s['anomaly_flags']) + ', Dist: ' + str(round(s['distance_km'], 2)) + 'km')