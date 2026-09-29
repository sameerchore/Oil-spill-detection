import pytest
from app.ais.live_stream import normalize_aisstream_message

def test_normalize_aisstream_position_report():
    sample_msg = {
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": 419000001,
            "ShipName": "TEST TANKER",
            "latitude": 18.95,
            "longitude": 72.15,
            "time_utc": "2026-09-02T12:00:00Z"
        },
        "Message": {
            "PositionReport": {
                "Sog": 12.4,
                "Cog": 135.0,
                "TrueHeading": 135,
                "NavigationalStatus": 0
            }
        }
    }
    record = normalize_aisstream_message(sample_msg)
    assert record is not None
    assert record["MMSI"] == 419000001
    assert record["VesselName"] == "TEST TANKER"
    assert record["LAT"] == 18.95
    assert record["LON"] == 72.15
    assert record["SOG"] == 12.4
    assert record["COG"] == 135.0

def test_normalize_aisstream_invalid_mmsi():
    sample_msg = {
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": None
        }
    }
    assert normalize_aisstream_message(sample_msg) is None
