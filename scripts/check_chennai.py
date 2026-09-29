from app.api.routes import _process_detection, load_metocean_data
from app.metocean.interpolator import MetoceanInterpolator
from app.drift.hindcast import run_hindcast

detection_fc = _process_detection(scene_id='scene_chennai_002')
slicks = [f for f in detection_fc['features'] if f['properties']['class_id'] == 2]
slick_geom = slicks[0]['geometry']

grid = load_metocean_data('scene_chennai_002', center_lat=13.25, center_lon=80.35, t_sat='2026-09-01T18:30:00Z')
interp = MetoceanInterpolator(grid)
res = run_hindcast(interp, slick_geom, '2026-09-01T18:30:00Z', hindcast_hours=24, num_particles=50, target_origin_hours=16)

print('Chennai origin time:', res['origin_time_iso'])
print('Chennai centroid at 16h:', round(res['origin_centroid'][0], 3), round(res['origin_centroid'][1], 3))