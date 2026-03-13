from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase


# =========================================================
# 数据库连接规则（优先级从高到低）
# 1) DATABASE_URL：云端 Supabase / Postgres
# 2) AUDIT_SCHEDULER_DB：本地手动指定 sqlite 文件
# 3) exe 运行：用户目录 ~/WNRH_AuditScheduler/audit_scheduler.db
# 4) 源码本地运行：项目根目录 ./data/audit_scheduler.db
#
# 说明：
# - Streamlit Cloud / 云端正式环境：请务必配置 DATABASE_URL
# - 本地开发：不配 DATABASE_URL 也可自动回退 sqlite
# =========================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_local_db_file() -> Path:
    # 1) 手动指定 sqlite 文件
    env = os.environ.get("AUDIT_SCHEDULER_DB", "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = (_project_root() / p).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 2) 打包 exe：固定到用户目录
    if getattr(sys, "frozen", False):
        base = Path.home() / "WNRH_AuditScheduler"
        base.mkdir(parents=True, exist_ok=True)
        return base / "audit_scheduler.db"

    # 3) 本地源码运行：固定到项目根目录 data/
    base = _project_root() / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base / "audit_scheduler.db"


def _normalize_database_url(url: str) -> str:
    """
    兼容一些平台给出的 postgres:// 前缀
    SQLAlchemy 推荐使用 postgresql://
    """
    url = (url or "").strip()
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def _resolve_database_config() -> tuple[str, Path | None, bool]:
    """
    返回:
    - DB_URL: str
    - DB_FILE: Path | None
    - IS_SQLITE: bool
    """
    # 云端正式环境优先使用外部数据库
    database_url = _normalize_database_url(os.environ.get("DATABASE_URL", ""))

    if database_url:
        return database_url, None, False

    # 回退到本地 sqlite（仅适合本地开发/测试）
    db_file = _resolve_local_db_file()
    db_url = f"sqlite:///{str(db_file).replace('\\', '/')}"
    return db_url, db_file, True


DB_URL, DB_FILE, IS_SQLITE = _resolve_database_config()

engine_kwargs = {
    "pool_pre_ping": True,
    "future": True,
}

if IS_SQLITE:
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DB_URL, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


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
    说明：
    1) PostgreSQL（Supabase）：
       - 由 Base.metadata.create_all(bind=engine) 负责建表
       - 此函数不做 sqlite 风格的 ALTER TABLE 轻量迁移
    2) SQLite：
       - 对旧库补齐缺失字段，兼容历史版本
    """
    if not IS_SQLITE:
        return

    if DB_FILE is None or not DB_FILE.exists():
        return

    conn = sqlite3.connect(str(DB_FILE))
    cur = conn.cursor()

    def table_exists(name: str) -> bool:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
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
        # --- auditors ---
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

        # --- tasks ---
        if table_exists("tasks"):
            add_col("tasks", "customer_name", "TEXT")
            add_col("tasks", "need_expert", "INTEGER DEFAULT 0")
            add_col("tasks", "required_headcount", "INTEGER DEFAULT 1")
            add_col("tasks", "required_days", "INTEGER DEFAULT 1")
            add_col("tasks", "specified_auditors", "TEXT")
            add_col("tasks", "preferred_experts", "TEXT")
            add_col("tasks", "required_gender", "TEXT DEFAULT '不限'")
            add_col("tasks", "end_date", "DATE")

        # --- schedules ---
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


def test_db_connection() -> tuple[bool, str]:
    """
    可选：用于页面诊断数据库连接是否正常
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, f"数据库连接正常：{DB_URL}"
    except Exception as e:
        return False, f"数据库连接失败：{e}"
