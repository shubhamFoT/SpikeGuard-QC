"""SQLite logging. PS5 explicitly asks for 'logging inspection results'."""
import csv
import json
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, frame_id INTEGER, defect_class TEXT,
  confidence REAL, bbox TEXT, area_frac REAL, severity TEXT, decision TEXT, rolling_rate REAL,
  latency_ms REAL, category TEXT, urgency TEXT, zone TEXT, loop_ma REAL, machine_source TEXT);
CREATE TABLE IF NOT EXISTS advisories(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, category TEXT, summary TEXT, likely_cause TEXT,
  action TEXT, urgency TEXT, source TEXT, wall_ms REAL, meta TEXT);
"""


class DB:
    def __init__(self, path="inspections.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def log_inspection(self, ev, latency_ms, fused, machine):
        with self.lock:
            self.conn.execute(
                "INSERT INTO inspections(ts,frame_id,defect_class,confidence,bbox,area_frac,severity,"
                "decision,rolling_rate,latency_ms,category,urgency,zone,loop_ma,machine_source) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ev.ts, ev.frame_id, ev.defect_class, ev.confidence, json.dumps(ev.bbox), ev.area_frac,
                 ev.severity, ev.decision, ev.rolling_defect_rate, latency_ms, fused.category,
                 fused.urgency, machine.zone if machine else "unavailable",
                 machine.loop_ma if machine else None, machine.source if machine else None))
            self.conn.commit()

    def log_advisory(self, ts, category, advice, meta):
        with self.lock:
            self.conn.execute(
                "INSERT INTO advisories(ts,category,summary,likely_cause,action,urgency,source,wall_ms,meta) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (ts, category, advice["summary"], advice["likely_cause"], advice["action"],
                 advice["urgency"], meta.get("source"), meta.get("wall_ms"), json.dumps(meta)))
            self.conn.commit()

    def recent(self, table, n=20):
        assert table in ("inspections", "advisories")
        with self.lock:
            rows = self.conn.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    def export_csv(self, table, path):
        assert table in ("inspections", "advisories")
        with self.lock:
            cur = self.conn.execute(f"SELECT * FROM {table} ORDER BY id")
            rows, cols = cur.fetchall(), [d[0] for d in cur.description]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        return len(rows)
