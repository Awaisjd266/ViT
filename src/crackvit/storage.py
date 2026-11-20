"""Optional local SQLite run registry; artifacts remain portable JSON/CSV files."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def registry(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, status TEXT NOT NULL, config_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS epochs (
            run_id TEXT NOT NULL REFERENCES runs(run_id), epoch INTEGER NOT NULL,
            train_loss REAL NOT NULL, val_loss REAL NOT NULL, val_f1 REAL,
            PRIMARY KEY(run_id, epoch)
        );
        CREATE INDEX IF NOT EXISTS runs_status ON runs(status);
        """)
        yield db


def register_run(path: Path, run_id: str, config: dict) -> None:
    with registry(path) as db:
        db.execute("INSERT INTO runs(run_id,status,config_json) VALUES(?,?,?)",
                   (run_id, "running", json.dumps(config)))


def record_epoch(path: Path, run_id: str, epoch: int, train_loss: float,
                 val_loss: float, val_f1: float | None) -> None:
    with registry(path) as db:
        db.execute("""INSERT INTO epochs VALUES(?,?,?,?,?)
                   ON CONFLICT(run_id,epoch) DO UPDATE SET
                   train_loss=excluded.train_loss, val_loss=excluded.val_loss,
                   val_f1=excluded.val_f1""",
                   (run_id, epoch, train_loss, val_loss, val_f1))


def finish_run(path: Path, run_id: str, status: str) -> None:
    with registry(path) as db:
        db.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))
