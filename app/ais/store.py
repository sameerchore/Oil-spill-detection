"""
SQLite Storage Engine for AIS Records with MarineCadastre Schema and Composite B-Tree Indexing.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Standard 16 MarineCadastre fields
MARINECADASTRE_COLUMNS = [
    "MMSI",
    "BaseDateTime",
    "LAT",
    "LON",
    "SOG",
    "COG",
    "Heading",
    "VesselName",
    "IMO",
    "CallSign",
    "VesselType",
    "Status",
    "Length",
    "Width",
    "Draft",
    "Cargo",
]


def parse_timestamp(value: Union[str, dt.datetime, int, float]) -> Tuple[str, float]:
    """
    Parse various timestamp formats into (iso_str, epoch_seconds).
    Standardizes on UTC ISO format: 'YYYY-MM-DDTHH:MM:SSZ'.
    """
    if isinstance(value, (int, float)):
        # Epoch seconds
        d = dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)
        return d.strftime("%Y-%m-%dT%H:%M:%SZ"), float(value)
    
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.strftime("%Y-%m-%dT%H:%M:%SZ"), value.timestamp()

    if isinstance(value, str):
        v = value.strip().replace(" ", "T")
        # Strip trailing Z or handle timezone offset
        try:
            if v.endswith("Z"):
                d = dt.datetime.fromisoformat(v[:-1]).replace(tzinfo=dt.timezone.utc)
            elif "+" in v[10:] or "-" in v[10:]:
                d = dt.datetime.fromisoformat(v).astimezone(dt.timezone.utc)
            else:
                d = dt.datetime.fromisoformat(v).replace(tzinfo=dt.timezone.utc)
            return d.strftime("%Y-%m-%dT%H:%M:%SZ"), d.timestamp()
        except Exception:
            # Fallback for common formats like 'YYYY-MM-DD HH:MM:SS'
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
                try:
                    d = dt.datetime.strptime(value.strip(), fmt).replace(tzinfo=dt.timezone.utc)
                    return d.strftime("%Y-%m-%dT%H:%M:%SZ"), d.timestamp()
                except ValueError:
                    pass
            raise ValueError(f"Unrecognized datetime format: {value}")
            
    raise TypeError(f"Cannot parse timestamp of type {type(value)}")


class AISStore:
    """
    SQLite-backed AIS database storage engine.
    Supports MarineCadastre 16-field schema and composite B-Tree indexing on (timestamp, lat, lon).
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:", auto_commit: bool = True):
        self.db_path = str(db_path)
        self.auto_commit = auto_commit
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self.init_db()

    def __enter__(self) -> AISStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def init_db(self) -> None:
        """Create schema and composite B-Tree indexes."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ais_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                MMSI INTEGER NOT NULL,
                BaseDateTime TEXT NOT NULL,
                timestamp_epoch REAL NOT NULL,
                LAT REAL NOT NULL,
                LON REAL NOT NULL,
                SOG REAL,
                COG REAL,
                Heading REAL,
                VesselName TEXT,
                IMO TEXT,
                CallSign TEXT,
                VesselType INTEGER,
                Status INTEGER,
                Length REAL,
                Width REAL,
                Draft REAL,
                Cargo INTEGER
            );
            """
        )

        # Composite B-Tree index on (BaseDateTime, LAT, LON) as required by spec
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ais_time_lat_lon 
            ON ais_records (BaseDateTime, LAT, LON);
            """
        )

        # Composite B-Tree index on (timestamp_epoch, LAT, LON) for high-performance range queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ais_epoch_lat_lon 
            ON ais_records (timestamp_epoch, LAT, LON);
            """
        )

        # Index on (MMSI, timestamp_epoch) for fast trajectory extraction
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ais_mmsi_epoch 
            ON ais_records (MMSI, timestamp_epoch);
            """
        )

        if self.auto_commit:
            self._conn.commit()

    def insert_record(self, record: Dict[str, Any]) -> None:
        """Insert a single AIS record."""
        self.insert_records([record])

    def insert_records(self, records: Union[List[Dict[str, Any]], pd.DataFrame]) -> int:
        """
        Batch insert records into the database.
        Accepts list of dictionaries or pandas DataFrame.
        """
        if isinstance(records, pd.DataFrame):
            records = records.to_dict(orient="records")

        if not records:
            return 0

        rows = []
        for r in records:
            # Flexible column mapping
            mmsi_raw = r.get("MMSI") or r.get("mmsi")
            if mmsi_raw is None:
                continue
            try:
                mmsi = int(mmsi_raw)
            except (ValueError, TypeError):
                continue

            dt_raw = r.get("BaseDateTime") or r.get("base_date_time") or r.get("datetime") or r.get("timestamp")
            if dt_raw is None:
                continue
            try:
                iso_time, epoch_time = parse_timestamp(dt_raw)
            except Exception:
                continue

            lat = float(r.get("LAT") if r.get("LAT") is not None else r.get("lat", 0.0))
            lon = float(r.get("LON") if r.get("LON") is not None else r.get("lon", 0.0))
            
            sog_val = r.get("SOG") if r.get("SOG") is not None else r.get("sog")
            sog = float(sog_val) if sog_val is not None and not pd.isna(sog_val) else None

            cog_val = r.get("COG") if r.get("COG") is not None else r.get("cog")
            cog = float(cog_val) if cog_val is not None and not pd.isna(cog_val) else None

            hdg_val = r.get("Heading") if r.get("Heading") is not None else r.get("heading")
            heading = float(hdg_val) if hdg_val is not None and not pd.isna(hdg_val) else None

            v_name = r.get("VesselName") or r.get("vessel_name") or r.get("name")
            v_name = str(v_name).strip() if v_name is not None and not pd.isna(v_name) else None

            imo = r.get("IMO") or r.get("imo")
            imo = str(imo).strip() if imo is not None and not pd.isna(imo) else None

            callsign = r.get("CallSign") or r.get("call_sign") or r.get("callsign")
            callsign = str(callsign).strip() if callsign is not None and not pd.isna(callsign) else None

            v_type_val = r.get("VesselType") if r.get("VesselType") is not None else r.get("vessel_type")
            try:
                v_type = int(v_type_val) if v_type_val is not None and not pd.isna(v_type_val) else None
            except (ValueError, TypeError):
                v_type = None

            status_val = r.get("Status") if r.get("Status") is not None else r.get("status")
            try:
                status = int(status_val) if status_val is not None and not pd.isna(status_val) else None
            except (ValueError, TypeError):
                status = None

            length_val = r.get("Length") if r.get("Length") is not None else r.get("length")
            length = float(length_val) if length_val is not None and not pd.isna(length_val) else None

            width_val = r.get("Width") if r.get("Width") is not None else r.get("width")
            width = float(width_val) if width_val is not None and not pd.isna(width_val) else None

            draft_val = r.get("Draft") if r.get("Draft") is not None else r.get("draft")
            draft = float(draft_val) if draft_val is not None and not pd.isna(draft_val) else None

            cargo_val = r.get("Cargo") if r.get("Cargo") is not None else r.get("cargo")
            try:
                cargo = int(cargo_val) if cargo_val is not None and not pd.isna(cargo_val) else None
            except (ValueError, TypeError):
                cargo = None

            rows.append((
                mmsi, iso_time, epoch_time, lat, lon, sog, cog, heading,
                v_name, imo, callsign, v_type, status, length, width, draft, cargo
            ))

        cursor = self._conn.cursor()
        cursor.executemany(
            """
            INSERT INTO ais_records (
                MMSI, BaseDateTime, timestamp_epoch, LAT, LON, SOG, COG, Heading,
                VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft, Cargo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        if self.auto_commit:
            self._conn.commit()

        return len(rows)

    def load_csv(self, csv_path: Union[str, Path], chunksize: int = 50000) -> int:
        """
        Load AIS data from a MarineCadastre standard CSV file in chunks.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"AIS CSV file not found: {csv_path}")

        total_inserted = 0
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            inserted = self.insert_records(chunk)
            total_inserted += inserted

        return total_inserted

    def query_bounding_box(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        start_time: Optional[Union[str, dt.datetime, float]] = None,
        end_time: Optional[Union[str, dt.datetime, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query records falling within a spatial bounding box and optional time window.
        """
        query = [
            "SELECT * FROM ais_records WHERE LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ?"
        ]
        params: List[Any] = [min_lat, max_lat, min_lon, max_lon]

        if start_time is not None:
            _, start_epoch = parse_timestamp(start_time)
            query.append("AND timestamp_epoch >= ?")
            params.append(start_epoch)

        if end_time is not None:
            _, end_epoch = parse_timestamp(end_time)
            query.append("AND timestamp_epoch <= ?")
            params.append(end_epoch)

        query.append("ORDER BY MMSI, timestamp_epoch ASC")
        sql = " ".join(query)

        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_vessel_records(
        self,
        mmsi: Union[int, str],
        start_time: Optional[Union[str, dt.datetime, float]] = None,
        end_time: Optional[Union[str, dt.datetime, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all time-sorted fixes for a specific vessel MMSI.
        """
        query = ["SELECT * FROM ais_records WHERE MMSI = ?"]
        params: List[Any] = [int(mmsi)]

        if start_time is not None:
            _, start_epoch = parse_timestamp(start_time)
            query.append("AND timestamp_epoch >= ?")
            params.append(start_epoch)

        if end_time is not None:
            _, end_epoch = parse_timestamp(end_time)
            query.append("AND timestamp_epoch <= ?")
            params.append(end_epoch)

        query.append("ORDER BY timestamp_epoch ASC")
        sql = " ".join(query)

        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_distinct_mmsi_in_window(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        start_time: Union[str, dt.datetime, float],
        end_time: Union[str, dt.datetime, float],
    ) -> List[int]:
        """
        Get list of unique MMSIs that have at least one record in the given spatiotemporal window.
        """
        _, start_epoch = parse_timestamp(start_time)
        _, end_epoch = parse_timestamp(end_time)

        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT MMSI FROM ais_records
            WHERE LAT BETWEEN ? AND ? 
              AND LON BETWEEN ? AND ? 
              AND timestamp_epoch BETWEEN ? AND ?
            ORDER BY MMSI;
            """,
            (min_lat, max_lat, min_lon, max_lon, start_epoch, end_epoch),
        )
        return [row[0] for row in cursor.fetchall()]

    def get_vessel_metadata(self, mmsi: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get the latest or most complete vessel metadata for an MMSI.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT MMSI, VesselName, IMO, CallSign, VesselType, Length, Width, Draft, Cargo
            FROM ais_records
            WHERE MMSI = ? AND (VesselName IS NOT NULL OR VesselType IS NOT NULL)
            ORDER BY timestamp_epoch DESC
            LIMIT 1;
            """,
            (int(mmsi),),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
            
        cursor.execute(
            "SELECT MMSI, VesselName, IMO, CallSign, VesselType, Length, Width, Draft, Cargo FROM ais_records WHERE MMSI = ? LIMIT 1;",
            (int(mmsi),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def count_records(self) -> int:
        """Return total count of AIS records in the database."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ais_records;")
        return cursor.fetchone()[0]
