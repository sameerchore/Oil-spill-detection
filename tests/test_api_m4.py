"""
Unit and Integration Tests for Worker M4:
FastAPI REST Backend, Route Endpoints, Static File Serving, and ReportLab PDF Generation.
"""

import io
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.report import generate_pdf_report
from app.config import SCENES_DIR, JOBS_DIR


@pytest.fixture
def client():
    return TestClient(app)


class TestM4ApiEndpoints:
    """Tests for all REST endpoints in app/api/routes.py and app/api/main.py."""

    def test_health_endpoint(self, client):
        """GET /api/health returns 200, online status, and offline air-gapped badge."""
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "online"
        assert data["air_gapped"] is True
        assert "AIR-GAPPED" in data["air_gapped_badge"]
        assert "device" in data
        assert data["ais_records_count"] > 0

    def test_scenes_catalog_endpoint(self, client):
        """GET /api/scenes lists pre-cached demo scenes with complete metadata."""
        res = client.get("/api/scenes")
        assert res.status_code == 200
        data = res.json()
        assert "scenes" in data
        assert len(data["scenes"]) >= 4
        scene_ids = [s["id"] for s in data["scenes"]]
        assert "scene_mumbai_001" in scene_ids
        assert "scene_clean_004" in scene_ids

    def test_detect_endpoint_json(self, client):
        """POST /api/detect runs 3-class segmentation and returns RFC 7946 GeoJSON."""
        res = client.post("/api/detect", json={"scene_id": "scene_mumbai_001"})
        assert res.status_code == 200
        geojson = res.json()
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) >= 1
        oil_slicks = [f for f in geojson["features"] if f["properties"].get("class_id") == 2]
        assert len(oil_slicks) >= 1
        assert oil_slicks[0]["properties"]["area_km2"] > 0

    def test_detect_endpoint_clean_scene(self, client):
        """POST /api/detect on clean ocean benchmark returns zero slicks."""
        res = client.post("/api/detect", json={"scene_id": "scene_clean_004"})
        assert res.status_code == 200
        geojson = res.json()
        oil_slicks = [f for f in geojson["features"] if f["properties"].get("class_id") == 2]
        assert len(oil_slicks) == 0

    def test_drift_endpoint(self, client):
        """POST /api/drift executes backward Lagrangian hindcast and forward 36h forecast."""
        slick_geom = {
            "type": "Polygon",
            "coordinates": [
                [[72.14, 18.94], [72.16, 18.94], [72.16, 18.96], [72.14, 18.96], [72.14, 18.94]]
            ],
        }
        res = client.post("/api/drift", json={
            "slick_polygon": slick_geom,
            "scene_id": "scene_mumbai_001",
            "t_sat": "2026-09-02T12:00:00Z",
            "hindcast_hours": 24,
            "forecast_hours": 36,
        })
        assert res.status_code == 200
        data = res.json()
        assert "hindcast" in data
        assert "forecast" in data
        assert "origin_zone_geojson" in data
        assert data["hindcast"]["origin_time_iso"] is not None
        assert data["forecast"]["total_drift_km"] >= 0

    def test_attribute_endpoint(self, client):
        """POST /api/attribute executes AIS join and multi-factor scoring."""
        res = client.post("/api/attribute", json={
            "origin_zone": [18.75, 71.85],
            "origin_time": "2026-09-01T18:00:00Z",
            "radius_km": 10.0,
            "window_hours": 3.0,
        })
        assert res.status_code == 200
        data = res.json()
        assert "suspects" in data
        assert len(data["suspects"]) > 0
        culprit = data["culprit"]
        assert culprit is not None
        assert culprit["rank"] == 1
        assert culprit["confidence_pct"] > 0

    def test_master_run_endpoint(self, client):
        """POST /api/run executes full end-to-end orchestration with timing traces."""
        res = client.post("/api/run", json={"scene_id": "scene_mumbai_001"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "completed"
        assert "job_id" in data
        assert "execution_trace" in data
        assert len(data["execution_trace"]) == 3
        assert data["total_duration_ms"] > 0
        assert data["detection"]["type"] == "FeatureCollection"

    def test_job_polling_and_report_download(self, client):
        """GET /api/jobs/{id} and GET /api/report/{id} return execution state and PDF bytes."""
        # Run job
        run_res = client.post("/api/run", json={"scene_id": "scene_mumbai_001"})
        job_id = run_res.json()["job_id"]

        # Poll job
        poll_res = client.get(f"/api/jobs/{job_id}")
        assert poll_res.status_code == 200
        assert poll_res.json()["status"] == "completed"

        # Download ReportLab PDF
        pdf_res = client.get(f"/api/report/{job_id}")
        assert pdf_res.status_code == 200
        assert pdf_res.headers["content-type"] == "application/pdf"
        assert f"TideTrace_Forensic_Dossier_{job_id}.pdf" in pdf_res.headers["content-disposition"]
        assert len(pdf_res.content) > 2000
        # Verify PDF magic bytes '%PDF-'
        assert pdf_res.content.startswith(b"%PDF-")

    def test_static_index_serving(self, client):
        """GET / serves the naval tactical console index.html."""
        res = client.get("/")
        assert res.status_code == 200
        assert "TIDETRACE // NAVAL FORENSIC CONSOLE" in res.text
        assert "leaflet.js" in res.text


class TestReportLabPDFGenerator:
    """Tests for law-enforcement binary PDF forensic dossier generator."""

    def test_pdf_generation_full_dossier(self):
        """Verify complete ReportLab compilation with tables, badges, and evidence."""
        dummy_data = {
            "job_id": "job_test_001",
            "scene_id": "scene_mumbai_001",
            "t_sat": "2026-09-02T12:00:00Z",
            "detection": {
                "properties": {
                    "scene_id": "scene_mumbai_001",
                    "num_oil_slicks": 1,
                    "total_oil_area_km2": 4.85,
                    "num_lookalikes": 1,
                    "sensor": "Sentinel-1 C-SAR",
                }
            },
            "drift": {
                "hindcast": {
                    "origin_time_iso": "2026-09-01T18:00:00Z",
                    "origin_window_start_iso": "2026-09-01T15:00:00Z",
                    "origin_window_end_iso": "2026-09-01T21:00:00Z",
                    "origin_lat": 18.75,
                    "origin_lon": 71.85,
                    "age_hours_proxy": 18,
                    "ensemble_spread_km": 6.2,
                    "total_drift_km": 32.4,
                },
                "forecast": {
                    "forecast_end_time_iso": "2026-09-04T00:00:00Z",
                    "total_drift_km": 42.8,
                    "dominant_heading_deg": 72.5,
                    "threatened_bbox": [72.1, 18.9, 72.9, 19.4],
                },
            },
            "attribution": {
                "suspects": [
                    {
                        "rank": 1,
                        "vessel_name": "MT OCEAN VOYAGER",
                        "mmsi": 419000001,
                        "vessel_type": "Crude Oil Tanker",
                        "distance_km": 0.62,
                        "closest_time": "2026-09-01T18:00:00Z",
                        "s_prox": 0.813,
                        "s_type": 1.000,
                        "s_traj": 1.000,
                        "s_beh": 0.950,
                        "s_total": 0.915,
                        "confidence_pct": 91.5,
                        "cpa_sog": 12.5,
                        "cpa_cog": 65.0,
                        "reasons": [
                            "CPA distance 0.62 km to backtracked origin center",
                            "High-risk vessel class: Crude Oil Tanker",
                            "Heading aligned within 15 deg of drift trajectory",
                            "CRITICAL: 52-min transponder silence across origin window",
                        ],
                    }
                ]
            },
        }

        pdf_bytes = generate_pdf_report(dummy_data)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 2000
        assert pdf_bytes.startswith(b"%PDF-")

    def test_pdf_generation_clean_scene(self):
        """Verify PDF compiles gracefully for clean true-negative scenes."""
        clean_data = {
            "job_id": "job_clean_004",
            "scene_id": "scene_clean_004",
            "detection": {"properties": {"num_oil_slicks": 0, "total_oil_area_km2": 0.0}},
            "drift": {"hindcast": {}, "forecast": {}},
            "attribution": {"suspects": []},
        }
        pdf_bytes = generate_pdf_report(clean_data)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF-")
