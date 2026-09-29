from app.api.routes import _process_detection, load_metocean_data
from app.metocean.interpolator import MetoceanInterpolator
from app.drift.hindcast import run_hindcast

detection_fc = _process_detection(scene_id='scene_mumbai_001')
slicks = [f for f in detection_fc['features'] if f['properties']['class_id'] == 2]
slick_geom = slicks[0]['geometry']

grid = load_metocean_data('scene_mumbai_001', center_lat=18.95, center_lon=72.15, t_sat='2026-09-02T12:00:00Z')
interp = MetoceanInterpolator(grid)
res = run_hindcast(interp, slick_geom, '2026-09-02T12:00:00Z', hindcast_hours=48, num_particles=50)

print('Slick particles initial spread at h=0:', res['hourly_history'][0]['spread_km'])
for h in res['hourly_history'][:22]:
    print('Hour ' + str(h['hour_back']) + ': spread=' + str(round(h['spread_km'], 2)) + ' km, time=' + h['time_iso'] + ', centroid=(' + str(round(h['centroid_lat'], 3)) + ', ' + str(round(h['centroid_lon'], 3)) + ')')