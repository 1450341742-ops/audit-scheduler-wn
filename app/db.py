from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# =========================================================
# ✅ 关键修复：数据库路径唯一化（彻底解决“保存后不生效/刷新又变回去”）
# 之前的问题：sqlite:///./audit_scheduler.db 依赖当前工作目录，导致写A读B
#
# 现在规则（优先级从高到低）：
# 1) 环境变量 AUDIT_SCHEDULER_DB 指定（绝对/相对都行）
# 2) 若是打包 exe（sys.frozen=True）：固定到用户目录 ~/WNRH_AuditScheduler/
# 3) 否则源码运行：固定到项目根目录 ./data/
# =========================================================

def _project_root() -> Path:
    # app/db.py 在 app/ 下，项目根目录是上一级
    return Path(__file__).resolve().parent.parent


def _resolve_db_file() -> Path:
    # 1) 手动指定（可选）
    env = os.environ.get("AUDIT_SCHEDULER_DB", "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = (_project_root() / p).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 2) 打包 exe：不要放 exe 目录（可能无写权限/临时目录）
    if getattr(sys, "frozen", False):
        base = Path.home() / "WNRH_AuditScheduler"
        base.mkdir(parents=True, exist_ok=True)
        return base / "audit_scheduler.db"

    # 3) 源码运行：固定到项目根目录 data/
    base = _project_root() / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base / "audit_scheduler.db"


DB_FILE = _resolve_db_file()
DB_URL = f"sqlite:///{str(DB_FILE).replace('\\', '/')}"


engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_schema():
    """SQLite 轻量迁移：为已有数据库补齐缺失字段（避免升级后录入/查询异常）。
    ✅ 注意：必须使用 DB_FILE（唯一绝对路径），不能用 os.getcwd()。
    """
    db_file = DB_FILE
    if not db_file.exists():
        return

    conn = sqlite3.connect(str(db_file))
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
