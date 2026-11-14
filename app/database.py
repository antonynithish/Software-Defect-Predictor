
from pathlib import Path
import sqlite3
import json
from datetime import datetime, timezone


ROOT_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT_DIR / "database" / "defect_predictions.db"


def get_connection():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def init_db():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                prediction TEXT NOT NULL,
                defective_probability REAL NOT NULL,
                threshold REAL NOT NULL,
                risk_category TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            )
        """)


def save_prediction(metrics, result):
    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO prediction_history (
                created_at,
                prediction,
                defective_probability,
                threshold,
                risk_category,
                metrics_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            created_at,
            result["prediction"],
            result["defective_probability"],
            result["threshold"],
            result["risk_category"],
            json.dumps(metrics),
        ))

        return cursor.lastrowid


def get_prediction_history(limit=50):
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    history = []

    for row in rows:
        item = dict(row)
        item["metrics"] = json.loads(
            item.pop("metrics_json")
        )
        history.append(item)

    return history
