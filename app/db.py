from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DB_URL = "sqlite:///./audit_scheduler.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
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
    """SQLite 轻量迁移：为已有数据库补齐缺失字段（避免升级后录入/查询 500）。
    说明：本项目不使用 alembic，因此用 PRAGMA + ALTER TABLE 做最小迁移。
    """
    import sqlite3, os, re

    url = DB_URL
    if not url.startswith("sqlite"):
        return

    m = re.match(r"sqlite:(/{3,4})(.+)$", url)
    if not m:
        return
    path_part = m.group(2)

    db_file = path_part
    if not os.path.isabs(db_file):
        db_file = os.path.abspath(os.path.join(os.getcwd(), db_file))

    if not os.path.exists(db_file):
        return

    conn = sqlite3.connect(db_file)
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

        # --- city_distances (历史版本可能没有唯一约束，不做破坏性迁移) ---
        conn.commit()
    finally:
        conn.close()
