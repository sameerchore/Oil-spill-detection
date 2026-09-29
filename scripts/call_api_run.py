import urllib.request, json
req = urllib.request.Request('http://127.0.0.1:8000/api/run', data=json.dumps({'scene_id': 'scene_mumbai_001'}).encode('utf-8'), headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req)
data = json.loads(resp.read().decode())
print('Job ID:', data.get('job_id'))
print('Duration ms:', data.get('total_duration_ms'))
print('Slicks:', len(data.get('detection', {}).get('features', [])))
print('Origin time:', data.get('drift', {}).get('hindcast', {}).get('origin_time_iso'))
print('Age hours proxy:', data.get('drift', {}).get('hindcast', {}).get('age_hours_proxy'))
attr = data.get('attribution', {})
suspects = attr.get('suspects', [])
print('Suspects count:', len(suspects))
if suspects:
    print('Culprit:', suspects[0]['vessel_name'], 'Rank #', suspects[0]['rank'], 'MMSI:', suspects[0]['mmsi'], 'Conf:', suspects[0]['confidence_pct'], '%')
    for s in suspects[:5]:
        print('  #' + str(s['rank']) + ': ' + s['vessel_name'] + ' - ' + str(s['confidence_pct']) + '% - ' + str(s['anomaly_flags']))