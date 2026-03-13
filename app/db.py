from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_local_db_file() -> Path:
    env = os.environ.get("AUDIT_SCHEDULER_DB", "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = (_project_root() / p).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    if getattr(sys, "frozen", False):
        base = Path.home() / "WNRH_AuditScheduler"
        base.mkdir(parents=True, exist_ok=True)
        return base / "audit_scheduler.db"

    base = _project_root() / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base / "audit_scheduler.db"


def _normalize_database_url(url: str) -> str:
    url = (url or "").strip()
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def _resolve_database_config() -> tuple[str, Path | None, bool]:
    database_url = _normalize_database_url(os.environ.get("DATABASE_URL", ""))

    if database_url:
        return database_url, None, False

    db_file = _resolve_local_db_file()
    db_url = f"sqlite:///{db_file.as_posix()}"
    return db_url, db_file, True


DB_URL, DB_FILE, IS_SQLITE = _resolve_database_config()

engine_kwargs = {
    "pool_pre_ping": True,
    "future": True,
}

if IS_SQLITE:
    engine_kwargs["connect_args"] = {"check_same_thread": False}

if IS_SQLITE:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["connect_args"] = {
        "connect_timeout": 10,
        "sslmode": "require",
    }

engine = create_engine(DB_URL, **engine_kwargs)
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_schema():
    """
    PostgreSQL:
        由 Base.metadata.create_all(bind=engine) 建表
    SQLite:
        对历史库做轻量补字段
    """
    if not IS_SQLITE:
        return

    if DB_FILE is None or not DB_FILE.exists():
        return

    conn = sqlite3.connect(str(DB_FILE))
    cur = conn.cursor()

    def table_exists(name: str) -> bool:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        return cur.fetchone() is not None

    def get_cols(table: str):
        cur.execute(f"PRAGMA table_info({table})")
        return [r[1] for r in cur.fetchall()]

    def add_col(table: str, col: str, ddl: str):
        cols = get_cols(table)
        if col in cols:
            return
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    try:
        if table_exists("auditors"):
            add_col("auditors", "gender", "TEXT DEFAULT '男'")
            add_col("auditors", "group_level", "TEXT DEFAULT 'B'")
            add_col("auditors", "can_lead_team", "INTEGER DEFAULT 0")
            add_col("auditors", "max_weekly_tasks", "INTEGER DEFAULT 1")
            add_col("auditors", "monthly_cases", "INTEGER DEFAULT 0")
            add_col("auditors", "travel_days", "INTEGER DEFAULT 0")
            add_col("auditors", "continuous_days", "INTEGER DEFAULT 0")
            add_col("auditors", "last_task_end_city", "TEXT")
            add_col("auditors", "last_task_end_date", "DATE")
            add_col("auditors", "status", "TEXT DEFAULT 'active'")

        if table_exists("tasks"):
            add_col("tasks", "customer_name", "TEXT")
            add_col("tasks", "need_expert", "INTEGER DEFAULT 0")
            add_col("tasks", "required_headcount", "INTEGER DEFAULT 1")
            add_col("tasks", "required_days", "INTEGER DEFAULT 1")
            add_col("tasks", "specified_auditors", "TEXT")
            add_col("tasks", "preferred_experts", "TEXT")
            add_col("tasks", "required_gender", "TEXT DEFAULT '不限'")
            add_col("tasks", "end_date", "DATE")

        if table_exists("schedules"):
            add_col("schedules", "end_date", "DATE")
            add_col("schedules", "travel_from_city", "TEXT")
            add_col("schedules", "travel_to_city", "TEXT")
            add_col("schedules", "distance_km", "REAL DEFAULT 0")
            add_col("schedules", "score", "REAL DEFAULT 0")
            add_col("schedules", "status", "TEXT DEFAULT 'confirmed'")

        conn.commit()
    finally:
        conn.close()


def test_db_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, f"数据库连接正常：{DB_URL}"
    except Exception as e:
        return False, str(e)
