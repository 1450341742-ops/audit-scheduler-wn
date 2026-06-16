from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_streamlit_secret(name: str) -> str:
    """Read a root-level Streamlit secret without making Streamlit mandatory."""
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
        return str(value).strip() if value is not None else ""
    except Exception:
        return ""


def _read_setting(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    if value is not None and str(value).strip():
        return str(value).strip()

    secret_value = _read_streamlit_secret(name)
    if secret_value:
        return secret_value

    return default


def _resolve_local_db_file() -> Path:
    configured_path = _read_setting("AUDIT_SCHEDULER_DB")
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = (_project_root() / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if getattr(sys, "frozen", False):
        base = Path.home() / "WNRH_AuditScheduler"
        base.mkdir(parents=True, exist_ok=True)
        return base / "audit_scheduler.db"

    base = _project_root() / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base / "audit_scheduler.db"


def _normalize_database_url(url: str) -> str:
    normalized = (url or "").strip()
    if normalized.startswith("postgres://"):
        normalized = normalized.replace("postgres://", "postgresql://", 1)
    return normalized


def _configured_remote_database_url() -> str:
    for setting_name in ("DATABASE_URL", "SUPABASE_DB_URL", "POSTGRES_URL"):
        value = _normalize_database_url(_read_setting(setting_name))
        if value:
            return value
    return ""


def _bool_setting(name: str, default: bool) -> bool:
    raw = _read_setting(name, "true" if default else "false").strip().lower()
    return raw not in {"0", "false", "no", "off", "disabled"}


def _is_sqlite_url(url: str) -> bool:
    return str(url or "").lower().startswith("sqlite:")


def _engine_kwargs(url: str) -> dict:
    kwargs: dict = {
        "pool_pre_ping": True,
        "future": True,
        "hide_parameters": True,
    }

    if _is_sqlite_url(url):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(
            {
                "pool_recycle": 300,
                "pool_timeout": 10,
            }
        )
        kwargs["connect_args"] = {
            "connect_timeout": 10,
            "sslmode": "require",
            "application_name": "wnrh_audit_scheduler",
        }

    return kwargs


def _create_engine(url: str) -> Engine:
    return create_engine(url, **_engine_kwargs(url))


def _test_engine_connection(candidate_engine: Engine) -> None:
    with candidate_engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def _safe_database_label(url: str) -> str:
    try:
        parsed = make_url(url)
        return parsed.render_as_string(hide_password=True)
    except Exception:
        return "configured database"


def _compact_error(exc: Exception) -> str:
    original = getattr(exc, "orig", None)
    message = str(original or exc).strip().replace("\n", " ")
    if len(message) > 500:
        message = message[:500] + "..."
    return f"{type(exc).__name__}: {message}"


PRIMARY_DB_URL = _configured_remote_database_url()
REMOTE_DB_CONFIGURED = bool(PRIMARY_DB_URL)
ALLOW_SQLITE_FALLBACK = _bool_setting("ALLOW_SQLITE_FALLBACK", True)
USING_SQLITE_FALLBACK = False
PRIMARY_DB_ERROR = ""

DB_FILE: Path | None = None

if PRIMARY_DB_URL:
    try:
        engine = _create_engine(PRIMARY_DB_URL)
        _test_engine_connection(engine)
        DB_URL = PRIMARY_DB_URL
        IS_SQLITE = False
    except Exception as exc:
        PRIMARY_DB_ERROR = _compact_error(exc)
        try:
            engine.dispose()
        except Exception:
            pass

        if not ALLOW_SQLITE_FALLBACK:
            raise

        DB_FILE = _resolve_local_db_file()
        DB_URL = f"sqlite:///{DB_FILE.as_posix()}"
        engine = _create_engine(DB_URL)
        IS_SQLITE = True
        USING_SQLITE_FALLBACK = True
        print(
            "WARNING: Remote database is unavailable; using temporary SQLite fallback. "
            f"Remote={_safe_database_label(PRIMARY_DB_URL)}; Error={PRIMARY_DB_ERROR}",
            file=sys.stderr,
        )
else:
    DB_FILE = _resolve_local_db_file()
    DB_URL = f"sqlite:///{DB_FILE.as_posix()}"
    engine = _create_engine(DB_URL)
    IS_SQLITE = True


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


def _ensure_auth_table_and_default_admin() -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    is_super_admin INTEGER NOT NULL DEFAULT 0,
                    allowed_pages_json TEXT,
                    created_at TEXT
                )
                """
            )
        )

        count = conn.execute(text("SELECT COUNT(*) FROM auth_users")).scalar() or 0
        if int(count) == 0:
            password_hash = hashlib.sha256("admin123".encode("utf-8")).hexdigest()
            conn.execute(
                text(
                    """
                    INSERT INTO auth_users (
                        username,
                        password_hash,
                        is_admin,
                        is_super_admin,
                        allowed_pages_json,
                        created_at
                    ) VALUES (
                        'admin',
                        :password_hash,
                        1,
                        1,
                        NULL,
                        CURRENT_TIMESTAMP
                    )
                    """
                ),
                {"password_hash": password_hash},
            )


def ensure_schema() -> None:
    """Create current tables and upgrade historical SQLite columns."""
    Base.metadata.create_all(bind=engine)

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

    def get_cols(table: str) -> list[str]:
        cur.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cur.fetchall()]

    def add_col(table: str, col: str, ddl: str) -> None:
        if col in get_cols(table):
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


def initialize_database() -> None:
    """Initialize business tables and authentication before the login page runs."""
    ensure_schema()
    _ensure_auth_table_and_default_admin()


def test_db_connection() -> tuple[bool, str]:
    try:
        _test_engine_connection(engine)
        if USING_SQLITE_FALLBACK:
            return (
                True,
                "远程数据库暂不可用，当前使用临时 SQLite；恢复 Supabase 后重启应用即可切回。",
            )
        database_type = "SQLite" if IS_SQLITE else "PostgreSQL"
        return True, f"{database_type} 数据库连接正常"
    except Exception as exc:
        return False, _compact_error(exc)
