
try:
    CITY_COORDS
except NameError:
    try:
        from app.seed_distances import CITY_COORDS as _CITY_COORDS_FALLBACK
        CITY_COORDS = _CITY_COORDS_FALLBACK
    except Exception:
        CITY_COORDS = {}

try:
    SEED_CITY_DISTANCES
except NameError:
    try:
        from app.seed_distances import SEED_CITY_DISTANCES as _SEED_CITY_DISTANCES_FALLBACK
        SEED_CITY_DISTANCES = _SEED_CITY_DISTANCES_FALLBACK
    except Exception:
        SEED_CITY_DISTANCES = []

from __future__ import annotations
import csv
import io
import json
import os
import re
import hashlib
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import calendar
import math
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError


# ---- defensive helper shim ----
try:
    ensure_extra_tables
except NameError:
    def ensure_extra_tables():
        return None


def parse_name_list(raw) -> list[str]:
    s = normalize_text(raw)
    if not s:
        return []
    for sep in ["，", "、", ";", "；", "/", "|", "\n", "\t"]:
        s = s.replace(sep, ",")
    out = []
    seen = set()
    for x in s.split(','):
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def get_task_attribute_map(task_ids: list[int] | None = None) -> dict[int, dict]:
    store = st.session_state.get("_task_attr_store", {}) or {}
    if not task_ids:
        return {int(k): dict(v) for k, v in store.items()}
    out = {}
    for task_id in task_ids:
        task_id = int(task_id)
        if task_id in store:
            out[task_id] = dict(store[task_id])
    return out


def get_task_attributes(task_id: int) -> dict:
    task_id = int(task_id)
    mp = get_task_attribute_map([task_id])
    return mp.get(task_id, {"task_id": task_id, "capital_type": "", "project_phase": "", "disease_area": ""})


def save_task_attributes(task_id: int, capital_type: str = "", project_phase: str = "", disease_area: str = ""):
    task_id = int(task_id)
    store = dict(st.session_state.get("_task_attr_store", {}) or {})
    store[task_id] = {
        "task_id": task_id,
        "capital_type": normalize_text(capital_type),
        "project_phase": normalize_text(project_phase),
        "disease_area": normalize_text(disease_area),
    }
    st.session_state["_task_attr_store"] = store
    return True

def _preset_or_other(value: str, options: list[str]):
    v = normalize_text(value)
    if not v:
        return "", ""
    if v in options:
        return v, ""
    return "其他（手填）", v


def _merge_preset_and_other(selected: str, other_text: str) -> str:
    selected = normalize_text(selected)
    other_text = normalize_text(other_text)
    if selected == "其他（手填）":
        return other_text
    return selected


def get_auditor_capacity_map(auditor_ids: list[int] | None = None) -> dict[int, dict]:
    ids = {int(v) for v in (auditor_ids or [])}
    out = {}
    with db_session() as db:
        q = db.query(Auditor)
        if ids:
            q = q.filter(Auditor.id.in_(list(ids)))
        auditors = q.all()
        for a in auditors:
            base = int(getattr(a, "monthly_cases", 0) or 0)
            max_m = base if base > 0 else 6
            min_m = max(0, min(4, max_m)) if base > 0 else 4
            if max_m < min_m:
                max_m = min_m
            out[int(a.id)] = {"auditor_id": int(a.id), "min_monthly_cases": int(min_m), "max_monthly_cases": int(max_m)}
    for aid in ids:
        out.setdefault(int(aid), {"auditor_id": int(aid), "min_monthly_cases": 4, "max_monthly_cases": 6})
    return out


def save_auditor_capacity_target(auditor_id: int, min_monthly_cases: int = 4, max_monthly_cases: int = 6):
    min_cases = max(0, int(min_monthly_cases or 0))
    max_cases = max(min_cases, int(max_monthly_cases or 0))
    with db_session() as db:
        obj = db.query(Auditor).filter(Auditor.id == int(auditor_id)).first()
        if not obj:
            return False, "未找到稽查员"
        obj.monthly_cases = int(max_cases)
        ok = safe_commit(db, f"更新稽查员标准院次#{auditor_id}")
        return (True, "已保存") if ok else (False, "保存失败")

def get_subperiod_progress_rows(period_type: str, year: int, period_value: int):
    rows = []
    if period_type == "monthly":
        start_d, end_d = _get_period_range(period_type, year, period_value)
        cur = start_d
        idx = 1
        while cur <= end_d:
            week_end = min(end_d, cur + timedelta(days=6))
            w_target = get_target_row("weekly", year, int(cur.isocalendar().week)).get("target_projects", 0)
            actual = 0
            with db_session() as db:
                task_rows = db.query(Task).filter(Task.start_date >= cur, Task.start_date <= week_end).all()
                actual = len({int(t.id) for t in task_rows})
            rows.append({"标签": f"第{idx}周", "目标院次": int(w_target or 0), "完成院次": int(actual or 0)})
            cur = week_end + timedelta(days=1)
            idx += 1
    elif period_type == "quarterly":
        start_month = (int(period_value) - 1) * 3 + 1
        for m in range(start_month, start_month + 3):
            target = get_target_row("monthly", year, m).get("target_projects", 0)
            _, _, actual, _, _ = get_progress_stats("monthly", year, m)
            rows.append({"标签": f"{m}月", "目标院次": int(target or 0), "完成院次": int(actual or 0)})
    elif period_type == "yearly":
        for m in range(1, 13):
            target = get_target_row("monthly", year, m).get("target_projects", 0)
            _, _, actual, _, _ = get_progress_stats("monthly", year, m)
            q = (m - 1) // 3 + 1
            rows.append({"标签": f"{m}月", "季度": f"Q{q}", "目标院次": int(target or 0), "完成院次": int(actual or 0)})
    return rows


def _pick_cn_font(size=18, bold=False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_calendar_png_bytes(year: int, month: int, events_by_day: dict, day_marks: dict):
    cell_w, cell_h = 230, 130
    margin_x, margin_y = 30, 30
    title_h, head_h = 80, 50
    width = margin_x * 2 + cell_w * 7
    height = margin_y * 2 + title_h + head_h + cell_h * 6 + 20
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_title = _pick_cn_font(34, bold=True)
    font_head = _pick_cn_font(18, bold=True)
    font_day = _pick_cn_font(18, bold=True)
    font_body = _pick_cn_font(14, bold=False)
    draw.text((margin_x, margin_y), f"{year}年{month}月 排班日历", font=font_title, fill="#1f2937")
    headers = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    y0 = margin_y + title_h
    for i, h in enumerate(headers):
        x = margin_x + i * cell_w
        draw.rectangle([x, y0, x + cell_w, y0 + head_h], outline="#d1d5db", width=1, fill="#f8fafc")
        draw.text((x + 10, y0 + 12), h, font=font_head, fill="#374151")
    month_start = date(year, month, 1)
    first_cell = month_start - timedelta(days=month_start.weekday())
    current = first_cell
    for r in range(6):
        for c in range(7):
            x = margin_x + c * cell_w
            y = y0 + head_h + r * cell_h
            fill = "#ffffff" if current.month == month else "#f9fafb"
            if events_by_day.get(current.isoformat()):
                fill = "#eef6ff"
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline="#d1d5db", width=1, fill=fill)
            draw.text((x + 8, y + 8), str(current.day), font=font_day, fill="#111827")
            mark = day_marks.get(current.isoformat())
            line_y = y + 34
            if mark:
                draw.text((x + 8, line_y), str(mark.get("label") or mark.get("type") or ""), font=font_body, fill="#16a34a")
                line_y += 20
            day_events = events_by_day.get(current.isoformat(), [])
            show_events = day_events[:2]
            for obj in show_events:
                txt = f"• {obj.get('project','')} {'、'.join(obj.get('persons', []))}"
                if len(txt) > 24:
                    txt = txt[:24] + "…"
                draw.text((x + 8, line_y), txt, font=font_body, fill="#1f2937")
                line_y += 18
            if len(day_events) > 2:
                draw.text((x + 8, line_y), f"还有{len(day_events)-2}项…", font=font_body, fill="#6b7280")
            current += timedelta(days=1)
    bio = BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def load_day_marks() -> list[dict]:
    return []


def get_part_time_staff_rows(active_only: bool = False) -> list[dict]:
    return []


def save_part_time_staff(name: str, base_city: str = '', note: str = '', is_active: bool = True):
    return False, '纯稳定生产版已关闭兼职库持久化，请使用内部稽查员排班'


def delete_part_time_staff(row_id: int):
    return False, '纯稳定生产版已关闭兼职库持久化' 

def get_direct_assignments(task_id: int) -> list[dict]:
    with db_session() as db:
        rows = (
            db.query(Schedule, Auditor)
            .join(Auditor, Auditor.id == Schedule.auditor_id)
            .filter(Schedule.task_id == int(task_id))
            .order_by(Schedule.start_date.asc(), Schedule.id.asc())
            .all()
        )
    out = []
    for s, a in rows:
        out.append({
            "id": int(getattr(s, "id")),
            "task_id": int(task_id),
            "auditor_id": int(getattr(a, "id")),
            "person_name": getattr(a, "name", ""),
            "is_part_time": False,
            "role": getattr(s, "role", "member"),
            "start_date": d2s(getattr(s, "start_date", None)),
            "end_date": d2s(getattr(s, "end_date", None)),
            "project_name": "",
            "notes": "",
            "created_at": "",
        })
    return out


def replace_direct_assignments(task_id: int, rows: list[dict]):
    with db_session() as db:
        task = db.query(Task).filter(Task.id == int(task_id)).first()
        if not task:
            return False, "任务不存在"
        db.query(Schedule).filter(Schedule.task_id == int(task_id)).delete()
        for r in rows or []:
            auditor_id = r.get("auditor_id")
            if not auditor_id:
                continue
            auditor = db.query(Auditor).filter(Auditor.id == int(auditor_id)).first()
            if not auditor:
                continue
            sd = safe_parse_date(r.get("start_date")) or task.start_date
            ed = safe_parse_date(r.get("end_date")) or task.end_date or task.start_date
            if ed < sd:
                ed = sd
            from_city = compute_from_city(auditor, task)
            km = get_distance_km(db, from_city, task.site_city)
            db.add(Schedule(
                task_id=int(task.id),
                auditor_id=int(auditor.id),
                role='leader' if str(r.get('role')) == 'leader' else 'member',
                start_date=sd,
                end_date=ed,
                travel_from_city=from_city,
                travel_to_city=task.site_city,
                distance_km=float(km),
                score=0.0,
                status='confirmed',
            ))
        ok = safe_commit(db, f'直录排班 task#{task_id}')
        return (True, '已直录到排班表') if ok else (False, '直录失败')


def save_direct_assignments_from_df(task_id: int, df_in):
    with db_session() as db:
        auditors = db.query(Auditor).all()
        task = db.query(Task).filter(Task.id == int(task_id)).first()
    if not task:
        return False, '任务不存在'
    auditor_name_to_id = {a.name: a.id for a in auditors}
    rows_out = []
    skipped_part_time = 0
    for _, r in pd.DataFrame(df_in).iterrows():
        person_name = normalize_text(r.get('人员姓名'))
        if not person_name:
            continue
        is_part_time = '兼职' in normalize_text(r.get('类型'))
        if is_part_time:
            skipped_part_time += 1
            continue
        auditor_id = auditor_name_to_id.get(person_name)
        if not auditor_id:
            continue
        sd = safe_parse_date(r.get('开始日期')) or task.start_date
        ed = safe_parse_date(r.get('结束日期')) or task.end_date or task.start_date
        rows_out.append({
            'auditor_id': auditor_id,
            'person_name': person_name,
            'is_part_time': False,
            'role': 'leader' if normalize_text(r.get('角色')) == '组长' else 'member',
            'start_date': sd,
            'end_date': ed,
            'project_name': normalize_text(r.get('项目名称')) or normalize_text(task.project_name),
            'notes': normalize_text(r.get('备注')),
        })
    ok, msg = replace_direct_assignments(int(task_id), rows_out)
    if skipped_part_time:
        msg += f'；已忽略{skipped_part_time}条兼职记录（纯稳定生产版不写兼职库）'
    return ok, msg


def sync_task_schedules_from_direct_assignments(task):
    return True, '已定项目人员已直接写入排班表，无需再次同步' 

def _get_period_range(period_type: str, year: int, period_value: int):
    year = int(year)
    period_value = int(period_value or 0)
    if period_type == 'weekly':
        from datetime import datetime as _dt
        start_d = _dt.fromisocalendar(year, max(1, min(53, period_value)), 1).date()
        end_d = _dt.fromisocalendar(year, max(1, min(53, period_value)), 7).date()
        return start_d, end_d
    if period_type == 'monthly':
        start_d = date(year, max(1, min(12, period_value)), 1)
        end_d = ((start_d.replace(day=28) + timedelta(days=4)).replace(day=1)) - timedelta(days=1)
        return start_d, end_d
    if period_type == 'quarterly':
        q = max(1, min(4, period_value))
        start_month = (q - 1) * 3 + 1
        start_d = date(year, start_month, 1)
        end_month = start_month + 2
        end_tmp = date(year, end_month, 28) + timedelta(days=4)
        end_d = end_tmp.replace(day=1) - timedelta(days=1)
        return start_d, end_d
    return date(year, 1, 1), date(year, 12, 31)


def get_target_row(period_type: str, year: int, period_value: int) -> dict:
    key = f"{period_type}:{int(year)}:{int(period_value or 0)}"
    store = dict(st.session_state.get("_progress_targets_store", {}) or {})
    if key in store:
        row = dict(store[key])
        row.setdefault("period_type", period_type)
        row.setdefault("year", int(year))
        row.setdefault("period_value", int(period_value or 0))
        row.setdefault("target_projects", 0)
        row.setdefault("target_staffing", 0)
        return row

    start_d, end_d = _get_period_range(period_type, int(year), int(period_value or 0))
    with db_session() as db:
        tasks = db.query(Task).filter(Task.start_date >= start_d, Task.start_date <= end_d).all()
    auto_target = len({int(t.id) for t in tasks})
    return {
        'period_type': period_type,
        'year': int(year),
        'period_value': int(period_value or 0),
        'target_projects': int(auto_target),
        'target_staffing': 0
    }


def save_target_row(period_type: str, year: int, period_value: int, target_projects: int, target_staffing: int):
    key = f"{period_type}:{int(year)}:{int(period_value or 0)}"
    store = dict(st.session_state.get("_progress_targets_store", {}) or {})
    store[key] = {
        'period_type': period_type,
        'year': int(year),
        'period_value': int(period_value or 0),
        'target_projects': int(target_projects or 0),
        'target_staffing': int(target_staffing or 0),
    }
    st.session_state["_progress_targets_store"] = store
    return True

def get_progress_stats(period_type: str, year: int, period_value: int):
    start_d, end_d = _get_period_range(period_type, year, period_value)
    with db_session() as db:
        tasks = db.query(Task).filter(Task.start_date >= start_d, Task.start_date <= end_d).order_by(Task.start_date.asc(), Task.id.asc()).all()
        schedules = db.query(Schedule).filter(Schedule.start_date <= end_d, Schedule.end_date >= start_d).all()
        actual_projects = len({int(t.id) for t in tasks})
        actual_staffing = len(schedules)
        staffing_map = {}
        for s in schedules:
            staffing_map[int(s.task_id)] = staffing_map.get(int(s.task_id), 0) + 1
        attr_map = get_task_attribute_map([int(t.id) for t in tasks]) if tasks else {}
        detail_rows = []
        for t in tasks:
            extra = attr_map.get(int(t.id), {})
            detail_rows.append({
                '任务ID': int(t.id),
                '项目名称': t.project_name or '',
                '客户': t.customer_name or '',
                '属性': extra.get('capital_type') or '',
                '分期': extra.get('project_phase') or '',
                '疾病领域': extra.get('disease_area') or '',
                '城市': t.site_city or '',
                '开始日期': d2s(t.start_date),
                '结束日期': d2s(t.end_date or t.start_date),
                '已完成院次': 1,
                '已安排人数': int(staffing_map.get(int(t.id), 0)),
                '需求人数': int(t.required_headcount or 0),
                '需求天数': int(t.required_days or 0),
            })
    return start_d, end_d, actual_projects, actual_staffing, detail_rows


def seed_city_distances_if_needed(db: Session):
    if db.query(CityDistance).count() > 0:
        return

    seen = set()
    for a, b, km in SEED_CITY_DISTANCES:
        a = str(a).strip()
        b = str(b).strip()
        if not a or not b or a == b:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        exists = db.query(CityDistance).filter(CityDistance.from_city == a, CityDistance.to_city == b).first()
        if exists:
            continue
        db.add(CityDistance(from_city=a, to_city=b, km=float(km)))
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            continue
    safe_commit(db, "初始化城市距离")


SEED_CITIES = [(name, latlon[0], latlon[1]) for name, latlon in CITY_COORDS.items()]


def seed_cities_if_needed(db: Session):
    if db.query(City).count() > 0:
        return
    for name, lat, lon in SEED_CITIES:
        nm = str(name).strip()
        if not nm:
            continue
        db.add(City(name=nm, lat=float(lat), lon=float(lon)))
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            continue
    safe_commit(db, "初始化城市坐标")


@st.cache_resource(show_spinner=False)
def initialize_app_once():
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    with db_session() as db:
        seed_city_distances_if_needed(db)
        seed_cities_if_needed(db)
    bootstrap_auth_users_if_needed()
    return True


# -------------------- 权限 --------------------
ALL_PAGES = [
    "经营看板",
    "智能排班",
    "批量排班",
    "稽查员管理",
    "任务管理",
    "指标统计",
    "兼职库",
    "城市距离",
    "城市坐标",
    "模板导入",
    "日历视图",
    "账号管理",
    "数据清理",
]
DEFAULT_NORMAL_PAGES = ["经营看板", "任务管理", "稽查员管理", "日历视图", "指标统计"]


def hash_password(password: str) -> str:
    return hashlib.sha256(str(password).encode("utf-8")).hexdigest()


def ensure_auth_table():
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

    if IS_SQLITE:
        with engine.begin() as conn:
            cols = conn.execute(text("PRAGMA table_info(auth_users)")).mappings().all()
            existing = {str(c.get("name")) for c in cols}

            if "is_super_admin" not in existing:
                conn.execute(text("ALTER TABLE auth_users ADD COLUMN is_super_admin INTEGER NOT NULL DEFAULT 0"))

            if "allowed_pages_json" not in existing:
                conn.execute(text("ALTER TABLE auth_users ADD COLUMN allowed_pages_json TEXT"))
    else:
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS is_super_admin INTEGER NOT NULL DEFAULT 0")
            )
            conn.execute(
                text("ALTER TABLE auth_users ADD COLUMN IF NOT EXISTS allowed_pages_json TEXT")
            )

def _bootstrap_seed_users() -> dict[str, str]:
    users = {}
    try:
        secret_users = st.secrets.get("auth_users", None)
        if secret_users:
            users = {str(k): str(v) for k, v in dict(secret_users).items()}
    except Exception:
        pass
    if not users:
        env_json = os.environ.get("AUTH_USERS_JSON", "").strip()
        if env_json:
            try:
                data = json.loads(env_json)
                if isinstance(data, dict):
                    users = {str(k): str(v) for k, v in data.items()}
            except Exception:
                pass
    if not users:
        users = {"admin": "admin123"}
    return users


def bootstrap_auth_users_if_needed():
    ensure_auth_table()
    with engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM auth_users")).scalar() or 0
        if int(count) > 0:
            row = conn.execute(
                text("SELECT username, is_super_admin FROM auth_users WHERE username='admin'")
            ).mappings().first()
            if row and int(row.get("is_super_admin", 0)) != 1:
                conn.execute(text("UPDATE auth_users SET is_admin=1, is_super_admin=1 WHERE username='admin'"))
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for username, password in _bootstrap_seed_users().items():
            clean_user = str(username).strip()
            if not clean_user:
                continue

            is_admin = 1 if clean_user == "admin" else 0
            is_super = 1 if clean_user == "admin" else 0
            allowed = None if is_admin else json.dumps(DEFAULT_NORMAL_PAGES, ensure_ascii=False)

            conn.execute(
                text(
                    """
                    INSERT INTO auth_users (username, password_hash, is_admin, is_super_admin, allowed_pages_json, created_at)
                    VALUES (:username, :password_hash, :is_admin, :is_super_admin, :allowed_pages_json, :created_at)
                    """
                ),
                {
                    "username": clean_user,
                    "password_hash": hash_password(str(password)),
                    "is_admin": is_admin,
                    "is_super_admin": is_super,
                    "allowed_pages_json": allowed,
                    "created_at": now,
                },
            )


def list_auth_users() -> list[dict]:
    ensure_auth_table()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT username, is_admin, is_super_admin, allowed_pages_json, created_at
                FROM auth_users
                ORDER BY is_super_admin DESC, is_admin DESC, username ASC
                """
            )
        ).mappings().all()
    return [dict(r) for r in rows]


def get_auth_user(username: str) -> Optional[dict]:
    ensure_auth_table()
    clean_user = str(username or "").strip()
    if not clean_user:
        return None
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT username, password_hash, is_admin, is_super_admin, allowed_pages_json, created_at
                FROM auth_users
                WHERE username = :username
                """
            ),
            {"username": clean_user},
        ).mappings().first()
    return dict(row) if row else None


def _normalize_pages(pages: list[str]) -> list[str]:
    seen = set()
    out = []
    for p in pages or []:
        p = str(p).strip()
        if not p:
            continue
        if p not in ALL_PAGES:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def get_user_allowed_pages(username: str) -> list[str]:
    u = get_auth_user(username)
    if not u:
        return DEFAULT_NORMAL_PAGES[:]
    if int(u.get("is_admin", 0)) == 1:
        return ALL_PAGES[:]
    raw = u.get("allowed_pages_json") or ""
    try:
        arr = json.loads(raw) if raw else []
        if isinstance(arr, list):
            pages = _normalize_pages(arr)
            return pages if pages else DEFAULT_NORMAL_PAGES[:]
    except Exception:
        pass
    return DEFAULT_NORMAL_PAGES[:]


def set_user_allowed_pages(username: str, pages: list[str]) -> tuple[bool, str]:
    ensure_auth_table()
    clean_user = str(username or "").strip()
    if not clean_user:
        return False, "账号不能为空"
    u = get_auth_user(clean_user)
    if not u:
        return False, "账号不存在"
    if int(u.get("is_admin", 0)) == 1:
        return False, "管理员账号默认全功能，无需设置可见板块"
    pages = _normalize_pages(pages)
    if not pages:
        return False, "至少勾选1个可见板块"
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE auth_users SET allowed_pages_json = :v WHERE username = :username"),
            {"v": json.dumps(pages, ensure_ascii=False), "username": clean_user},
        )
    return True, "已保存可见板块"


def create_auth_user(username: str, password: str, is_admin: bool = False, is_super_admin: bool = False) -> tuple[bool, str]:
    ensure_auth_table()
    clean_user = str(username or "").strip()
    if not clean_user:
        return False, "账号不能为空"
    if len(clean_user) < 3:
        return False, "账号至少3位"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", clean_user):
        return False, "账号仅支持字母、数字、下划线、点、短横线"
    if len(str(password or "")) < 6:
        return False, "密码至少6位"
    if get_auth_user(clean_user):
        return False, "该账号已存在"

    if is_super_admin:
        is_admin = True

    allowed = None if is_admin else json.dumps(DEFAULT_NORMAL_PAGES, ensure_ascii=False)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO auth_users (username, password_hash, is_admin, is_super_admin, allowed_pages_json, created_at)
                VALUES (:username, :password_hash, :is_admin, :is_super_admin, :allowed_pages_json, :created_at)
                """
            ),
            {
                "username": clean_user,
                "password_hash": hash_password(password),
                "is_admin": 1 if is_admin else 0,
                "is_super_admin": 1 if is_super_admin else 0,
                "allowed_pages_json": allowed,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
    return True, "新增账号成功"


def update_auth_password(username: str, new_password: str) -> tuple[bool, str]:
    ensure_auth_table()
    clean_user = str(username or "").strip()
    if not clean_user:
        return False, "账号不能为空"
    if len(str(new_password or "")) < 6:
        return False, "新密码至少6位"
    if not get_auth_user(clean_user):
        return False, "账号不存在"
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE auth_users SET password_hash = :password_hash WHERE username = :username"),
            {"username": clean_user, "password_hash": hash_password(new_password)},
        )
    return True, "密码修改成功"


def delete_auth_user(username: str, current_user: str) -> tuple[bool, str]:
    ensure_auth_table()
    clean_user = str(username or "").strip()
    if clean_user == "admin":
        return False, "默认管理员 admin 不允许删除"
    if clean_user == str(current_user or "").strip():
        return False, "不能删除当前登录账号"
    if not get_auth_user(clean_user):
        return False, "账号不存在"
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM auth_users WHERE username = :username"), {"username": clean_user})
    return True, "账号已删除"


def check_login(username: str, password: str) -> bool:
    user = get_auth_user(username)
    if not user:
        return False
    return str(user.get("password_hash")) == hash_password(str(password))


initialize_app_once()


def render_login():
    st.title(APP_NAME)
    st.subheader("账号密码登录")
    st.caption("首次使用默认主管理员：admin / admin123")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("账号")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", type="primary")
    if submitted:
        if check_login(username, password):
            user = get_auth_user(username)
            st.session_state["logged_in"] = True
            st.session_state["login_user"] = str(username).strip()
            st.session_state["is_admin"] = bool(int(user.get("is_admin", 0))) if user else False
            st.session_state["is_super_admin"] = bool(int(user.get("is_super_admin", 0))) if user else False
            st.session_state["allowed_pages"] = get_user_allowed_pages(str(username).strip())
            st.rerun()
        else:
            st.error("账号或密码错误")
    st.stop()


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False
if "is_super_admin" not in st.session_state:
    st.session_state["is_super_admin"] = False
if "allowed_pages" not in st.session_state:
    st.session_state["allowed_pages"] = DEFAULT_NORMAL_PAGES[:]
if "data_version" not in st.session_state:
    st.session_state["data_version"] = 0

if not st.session_state["logged_in"]:
    render_login()

STATUS_MAP = {"在岗": "active", "请假": "leave", "冻结": "frozen"}
STATUS_MAP_REV = {v: k for k, v in STATUS_MAP.items()}
BOOL_TRUE = {"是", "Y", "y", "yes", "YES", "True", "true", "1", "是/yes"}


def ics_escape(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def build_ics_events(db: Session, auditor_id: int | None = None):
    q = db.query(Schedule).options(joinedload(Schedule.task), joinedload(Schedule.auditor)).order_by(Schedule.id.desc())
    if auditor_id:
        q = q.filter(Schedule.auditor_id == auditor_id)
    sch = q.all()
    events = []
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for s in sch:
        a = s.auditor
        t = s.task
        if not a or not t:
            continue
        start = datetime.combine(t.start_date, datetime.min.time()).replace(hour=9)
        actual_end = t.end_date or (t.start_date + timedelta(days=max(1, int(t.required_days or 1)) - 1))
        if actual_end < t.start_date:
            actual_end = t.start_date
        end_exclusive = datetime.combine(actual_end + timedelta(days=1), datetime.min.time()).replace(hour=18)
        uid = f"wnrh-{s.id}@scheduler"
        summary = f"{t.project_name}｜{t.site_city}｜{s.role}"
        desc = f"客户:{t.customer_name or ''}\n人数:{t.required_headcount} 天数:{t.required_days}\n负责人/成员:{a.name}"
        events.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{now}",
                f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
                f"DTEND:{end_exclusive.strftime('%Y%m%dT%H%M%S')}",
                f"SUMMARY:{ics_escape(summary)}",
                f"DESCRIPTION:{ics_escape(desc)}",
                "END:VEVENT",
            ]
        )
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//WNRH Scheduler//CN",
        "CALSCALE:GREGORIAN",
        "X-WR-CALNAME:万宁睿和排班",
        *events,
        "END:VCALENDAR",
    ]
    return "\r\n".join(lines).encode("utf-8")


@st.cache_data(show_spinner=False, ttl=120)
def build_ics_events_cached(data_version: int, auditor_id: int | None = None):
    with db_session() as db:
        return build_ics_events(db, auditor_id=auditor_id)


@st.cache_data(show_spinner=False, ttl=120)
def get_auditors_for_ui(data_version: int):
    with db_session() as db:
        auditors = db.query(Auditor).order_by(Auditor.name.asc()).all()
        return [{"id": int(a.id), "name": a.name or ""} for a in auditors]


@st.cache_data(show_spinner=False, ttl=120)
def get_calendar_payload(data_version: int, year: int, month: int, auditor_id: int | None = None):
    month_start = date(int(year), int(month), 1)
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    month_end = next_month - timedelta(days=1)

    with db_session() as db:
        all_schedules = (
            db.query(Schedule)
            .options(joinedload(Schedule.task), joinedload(Schedule.auditor))
            .filter(Schedule.start_date <= month_end, Schedule.end_date >= month_start)
            .order_by(Schedule.start_date.asc())
            .all()
        )

    direct_rows_raw = []
    globals().get("ensure_extra_tables", lambda: None)()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT da.id, da.task_id, da.auditor_id, da.person_name, da.is_part_time, da.role, da.start_date, da.end_date, da.notes,
                       t.project_name, t.site_city
                FROM direct_assignments da
                LEFT JOIN tasks t ON da.task_id = t.id
                WHERE da.start_date <= :month_end AND da.end_date >= :month_start
                ORDER BY da.start_date ASC, da.person_name ASC
            """),
            {"month_start": d2s(month_start), "month_end": d2s(month_end)},
        ).mappings().all()
        direct_rows_raw = [dict(r) for r in rows]

    direct_task_ids = {int(r["task_id"]) for r in direct_rows_raw}
    all_schedules_rows = []
    for s in all_schedules:
        if int(s.task_id) in direct_task_ids:
            continue
        if auditor_id and s.auditor_id != auditor_id:
            continue
        all_schedules_rows.append(
            {
                "id": s.id,
                "auditor_id": s.auditor_id,
                "auditor_name": (s.auditor.name if s.auditor else ""),
                "task_id": s.task_id,
                "project_name": (s.task.project_name if s.task else ""),
                "site_city": (s.task.site_city if s.task else ""),
                "role": s.role,
                "start_date": s.start_date,
                "end_date": s.end_date,
                "travel_from_city": s.travel_from_city,
                "travel_to_city": s.travel_to_city,
                "distance_km": float(s.distance_km or 0),
                "source": "schedule",
            }
        )

    direct_rows = []
    for r in direct_rows_raw:
        if auditor_id and r.get("auditor_id") and int(r.get("auditor_id")) != int(auditor_id):
            continue
        direct_rows.append(
            {
                "id": r.get("id"),
                "auditor_id": r.get("auditor_id"),
                "auditor_name": r.get("person_name") or "",
                "task_id": r.get("task_id"),
                "project_name": r.get("project_name") or "",
                "site_city": r.get("site_city") or "",
                "role": r.get("role") or "member",
                "start_date": safe_parse_date(r.get("start_date")),
                "end_date": safe_parse_date(r.get("end_date")),
                "travel_from_city": "",
                "travel_to_city": r.get("site_city") or "",
                "distance_km": 0.0,
                "source": "direct",
            }
        )

    merged_rows = all_schedules_rows + direct_rows
    day_marks = {it.get("date"): it for it in load_day_marks() if it.get("date", "")[:7] == month_start.strftime("%Y-%m")}
    events_by_day = {}
    for s in merged_rows:
        cur = s.get("start_date")
        end_d = s.get("end_date")
        while cur and end_d and cur <= end_d:
            day_iso = cur.isoformat()
            bucket = events_by_day.setdefault(day_iso, {})
            task_key = int(s.get("task_id"))
            item = bucket.get(task_key)
            if item is None:
                item = {"project": s.get("project_name") or f"任务#{s.get('task_id')}", "task_id": s.get("task_id"), "persons": [], "city": s.get("site_city") or ""}
                bucket[task_key] = item
            nm = s.get("auditor_name") or ""
            if nm and nm not in item["persons"]:
                item["persons"].append(nm)
            cur += timedelta(days=1)
    events_by_day = {k: list(v.values()) for k, v in events_by_day.items()}
    return {"month_start": month_start, "month_end": month_end, "merged_rows": merged_rows, "events_by_day": events_by_day, "day_marks": day_marks}

def update_auditor_record(
    auditor_id: int,
    name: str,
    gender: str,
    group_level: str,
    can_lead_team: bool,
    base_city: str,
    max_weekly_tasks: int,
    status_cn: str,
    monthly_cases: int,
    travel_days: int,
    continuous_days: int,
    last_task_end_city: str,
    last_task_end_date,
):
    with db_session() as db:
        obj = db.query(Auditor).filter(Auditor.id == int(auditor_id)).first()
        if not obj:
            st.error("未找到对应稽查员记录")
            return False

        obj.name = normalize_text(name)
        obj.gender = normalize_text(gender) or "女"
        obj.group_level = normalize_text(group_level) or "B"
        obj.can_lead_team = bool(can_lead_team)
        obj.base_city = normalize_text(base_city)
        obj.max_weekly_tasks = int(max_weekly_tasks or 0)
        obj.status = STATUS_MAP.get(normalize_text(status_cn), "active")
        obj.monthly_cases = int(monthly_cases or 0)
        obj.travel_days = int(travel_days or 0)
        obj.continuous_days = int(continuous_days or 0)
        obj.last_task_end_city = normalize_text(last_task_end_city) or None

        parsed_date = safe_parse_date(last_task_end_date)
        if parsed_date is not None:
            obj.last_task_end_date = parsed_date

        if not obj.name:
            st.error("姓名不能为空")
            db.rollback()
            return False
        if not obj.base_city:
            st.error("常驻城市不能为空")
            db.rollback()
            return False

        return safe_commit(db, f"更新稽查员#{auditor_id}")


def delete_auditor_record(auditor_id: int):
    with db_session() as db:
        obj = db.query(Auditor).filter(Auditor.id == int(auditor_id)).first()
        if not obj:
            st.error("未找到对应稽查员记录")
            return False
        db.query(Schedule).filter(Schedule.auditor_id == int(auditor_id)).delete()
        db.delete(obj)
        return safe_commit(db, f"删除稽查员#{auditor_id}")


def update_task_record(
    task_id: int,
    project_name: str,
    customer_name: str,
    need_expert: bool,
    required_headcount: int,
    required_days: int,
    required_gender: str,
    specified_auditors: str,
    preferred_experts: str,
    site_city: str,
    start_date_value,
    end_date_value,
    capital_type: str = "",
    project_phase: str = "",
    disease_area: str = "",
):
    with db_session() as db:
        obj = db.query(Task).filter(Task.id == int(task_id)).first()
        if not obj:
            st.error("未找到对应任务记录")
            return False

        sd = safe_parse_date(start_date_value)
        ed = safe_parse_date(end_date_value)

        if sd is None:
            st.error("开始日期无效")
            return False

        if ed is None:
            ed = sd + timedelta(days=max(1, int(required_days or 1)) - 1)

        if ed < sd:
            ed = sd + timedelta(days=max(1, int(required_days or 1)) - 1)

        if ed < sd:
            st.error("结束日期不能早于开始日期")
            return False

        final_days = max(1, int(required_days or ((ed - sd).days + 1)))

        obj.project_name = normalize_text(project_name)
        obj.customer_name = normalize_text(customer_name) or None
        obj.need_expert = bool(need_expert)
        obj.required_headcount = max(1, int(required_headcount or 1))
        obj.required_days = final_days
        obj.required_gender = normalize_text(required_gender) or "不限"
        obj.specified_auditors = normalize_text(specified_auditors) or None
        obj.preferred_experts = normalize_text(preferred_experts) or None
        obj.site_city = normalize_text(site_city)
        obj.start_date = sd
        obj.end_date = ed

        if not obj.project_name:
            st.error("项目名称不能为空")
            db.rollback()
            return False
        if not obj.site_city:
            st.error("中心城市不能为空")
            db.rollback()
            return False

        schedules = db.query(Schedule).filter(Schedule.task_id == int(task_id)).all()
        for s in schedules:
            s.start_date = sd
            s.end_date = ed
            s.travel_to_city = obj.site_city

        ok = safe_commit(db, f"更新任务#{task_id}")
        if ok:
            save_task_attributes(int(task_id), capital_type=capital_type, project_phase=project_phase, disease_area=disease_area)
        return ok


def delete_task_record(task_id: int):
    with db_session() as db:
        obj = db.query(Task).filter(Task.id == int(task_id)).first()
        if not obj:
            st.error("未找到对应任务记录")
            return False
        db.query(Schedule).filter(Schedule.task_id == int(task_id)).delete()
        db.delete(obj)
        return safe_commit(db, f"删除任务#{task_id}")


def assign_team_to_task(db: Session, task: Task, leader_id: int, member_ids: list[int]):
    if db.query(Schedule).filter(Schedule.task_id == task.id).count() > 0:
        return False, "该任务已存在排班记录，不能重复排班"

    start_date = task.start_date
    end_date = task.end_date or (task.start_date + timedelta(days=max(1, int(task.required_days or 1)) - 1))
    selected_ids = [int(leader_id)] + [int(x) for x in member_ids if int(x) != int(leader_id)]

    for aid in selected_ids:
        existing = db.query(Schedule).filter(Schedule.auditor_id == aid).all()
        for s in existing:
            if not (end_date < s.start_date or s.end_date < start_date):
                return False, f"稽查员#{aid} 与已有任务时间冲突"

        auditor = db.query(Auditor).filter(Auditor.id == aid).first()
        if auditor and auditor.last_task_end_date and auditor.last_task_end_date >= start_date:
            return False, f"稽查员 {auditor.name} 的上次结束日期与本次开始日期冲突"

    def add_schedule(auditor_id: int, role: str):
        auditor = db.query(Auditor).filter(Auditor.id == auditor_id).first()
        if not auditor:
            return

        from_city = compute_from_city(auditor, task)
        km = get_distance_km(db, from_city, task.site_city)

        db.add(
            Schedule(
                task_id=task.id,
                auditor_id=auditor.id,
                role=role,
                start_date=start_date,
                end_date=end_date,
                travel_from_city=from_city,
                travel_to_city=task.site_city,
                distance_km=float(km),
                score=0.0,
                status="confirmed",
            )
        )

        auditor.monthly_cases = int(auditor.monthly_cases or 0) + 1
        days = (end_date - start_date).days + 1
        auditor.travel_days = int(auditor.travel_days or 0) + max(0, days)
        auditor.continuous_days = max(int(auditor.continuous_days or 0), days)
        auditor.last_task_end_city = task.site_city
        auditor.last_task_end_date = end_date

    add_schedule(int(leader_id), "leader")
    for mid in member_ids:
        if int(mid) != int(leader_id):
            add_schedule(int(mid), "member")

    return True, "ok"


def run_batch_schedule(db: Session, d1: date, d2: date, mode: str = "greedy"):
    if d2 < d1:
        d1, d2 = d2, d1

    scheduled_task_ids = {tid for (tid,) in db.query(Schedule.task_id).distinct().all()}
    tasks = db.query(Task).filter(Task.start_date >= d1, Task.start_date <= d2).all()
    tasks = [t for t in tasks if t.id not in scheduled_task_ids]
    tasks.sort(key=lambda t: (0 if t.need_expert else 1, -int(t.required_headcount or 1), t.start_date))

    auditors = db.query(Auditor).all()
    report = {"assigned": [], "skipped": [], "batch_week_counts": {}}

    for t in tasks:
        schedules_all = db.query(Schedule).all()
        candidates = build_candidates(db, t, auditors, schedules_all)
        team = propose_team(t, candidates)

        if mode == "optimized" and candidates:
            avg_cases = float(sum(int(a.monthly_cases or 0) for a in auditors) / max(1, len(auditors)))
            leader_pool = [c for c in candidates if c.can_lead_team]
            if t.need_expert:
                leader_pool = [c for c in leader_pool if c.group_level == "A"]
            leader_pool = leader_pool[:5]
            member_pool_all = candidates[:12]
            auditor_lookup = {a.id: a for a in auditors}
            best_team = None
            best_obj = None

            from app.scheduler import TeamProposal

            for leader in leader_pool:
                member_pool = [c for c in member_pool_all if c.auditor_id != leader.auditor_id]
                need_n = max(0, int(t.required_headcount or 1) - 1)

                if need_n == 0:
                    cand_team = TeamProposal(
                        leader=leader,
                        members=[],
                        team_score=leader.score,
                        notes="optimized-single",
                    )
                    obj = team_objective(cand_team, auditor_lookup, avg_cases, report["batch_week_counts"])
                    if best_obj is None or obj < best_obj:
                        best_obj, best_team = obj, cand_team
                    continue

                base_members = member_pool[:need_n]
                if len(base_members) < need_n:
                    continue

                cand_team = TeamProposal(
                    leader=leader,
                    members=base_members,
                    team_score=leader.score + sum(m.score for m in base_members) / max(1, len(base_members)),
                    notes="optimized",
                )
                obj = team_objective(cand_team, auditor_lookup, avg_cases, report["batch_week_counts"])
                if best_obj is None or obj < best_obj:
                    best_obj, best_team = obj, cand_team

            if best_team:
                team = best_team

        if not team:
            report["skipped"].append({"task_id": t.id, "project": t.project_name, "reason": "无可用团队"})
            continue

        leader_id = int(team.leader.auditor_id)
        member_ids = [int(m.auditor_id) for m in team.members]

        ok, msg = assign_team_to_task(db, t, leader_id, member_ids)
        if not ok:
            db.rollback()
            report["skipped"].append({"task_id": t.id, "project": t.project_name, "reason": msg})
            continue

        for aid in [leader_id] + member_ids:
            report["batch_week_counts"][aid] = int(report["batch_week_counts"].get(aid, 0)) + 1

        if not safe_commit(db, context=f"批量排班 commit：task#{t.id} {t.project_name}"):
            report["skipped"].append({"task_id": t.id, "project": t.project_name, "reason": "数据库写入失败"})
            continue

        report["assigned"].append(
            {
                "task_id": t.id,
                "project": t.project_name,
                "leader": team.leader.auditor_name,
                "members": [m.auditor_name for m in team.members],
            }
        )

    return report


def _candidate_to_dict(c):
    return {
        "auditor_id": int(c.auditor_id),
        "auditor_name": c.auditor_name,
        "group_level": c.group_level,
        "can_lead_team": bool(c.can_lead_team),
        "from_city": c.from_city,
        "km": float(c.km),
        "score": float(c.score),
        "explain": c.explain,
    }


def _dict_to_candidate(d):
    return SimpleNamespace(**d)


def _team_to_dict(team):
    if not team:
        return None
    return {
        "leader": _candidate_to_dict(team.leader),
        "members": [_candidate_to_dict(m) for m in getattr(team, "members", [])],
        "team_score": float(getattr(team, "team_score", 0.0)),
        "notes": getattr(team, "notes", ""),
    }


def _dict_to_team(d):
    if not d:
        return None
    return SimpleNamespace(
        leader=_dict_to_candidate(d["leader"]),
        members=[_dict_to_candidate(x) for x in d.get("members", [])],
        team_score=d.get("team_score", 0.0),
        notes=d.get("notes", ""),
    )


@st.cache_data(show_spinner=False, ttl=120)
@st.cache_data(show_spinner=False, ttl=120)
def get_tasks_for_ui(data_version: int):
    with db_session() as db:
        tasks = db.query(Task).order_by(Task.id.desc()).all()
        rows = []
        for t in tasks:
            rows.append({
                "id": int(t.id),
                "label": f"#{t.id} {t.project_name}｜{t.site_city}｜{d2s(t.start_date)}｜{t.required_days}天｜{t.required_headcount}人",
            })
        return rows


@st.cache_data(show_spinner=False, ttl=120)
def get_recent_schedule_rows(data_version: int, limit: int = 120):
    with db_session() as db:
        schedules_recent = (
            db.query(Schedule)
            .options(joinedload(Schedule.task), joinedload(Schedule.auditor))
            .order_by(Schedule.id.desc())
            .limit(limit)
            .all()
        )
        rows = []
        for s in schedules_recent:
            rows.append(
                {
                    "ID": s.id,
                    "任务": f"#{s.task_id} {(s.task.project_name if s.task else '')}",
                    "人员": f"#{s.auditor_id} {(s.auditor.name if s.auditor else '')} ({(s.auditor.group_level if s.auditor else '')})",
                    "角色": s.role,
                    "时间": f"{d2s(s.start_date)} ~ {d2s(s.end_date)}",
                    "路线": f"{s.travel_from_city} → {s.travel_to_city}",
                    "km": round(float(s.distance_km or 0), 1),
                }
            )
        return rows


@st.cache_data(show_spinner=False, ttl=120)
def get_recommendation_payload(task_id: int, data_version: int):
    with db_session() as db:
        task = db.query(Task).filter(Task.id == int(task_id)).first()
        if not task:
            return {"task_id": int(task_id), "candidates": [], "team": None, "error": "任务不存在"}

        auditors = db.query(Auditor).all()
        schedules_all = db.query(Schedule).all()
        candidates = build_candidates(db, task, auditors, schedules_all)
        team = propose_team(task, candidates)
        return {
            "task_id": int(task_id),
            "candidates": [_candidate_to_dict(c) for c in candidates[:25]],
            "team": _team_to_dict(team),
            "error": None if team else "无可用团队方案",
        }


# -------------------- 侧边栏 --------------------
st.sidebar.title(APP_NAME)
st.sidebar.caption(f"当前用户：{st.session_state.get('login_user', '')}")

if st.sidebar.button("退出登录", key="logout_btn"):
    st.session_state["logged_in"] = False
    st.session_state["is_admin"] = False
    st.session_state["is_super_admin"] = False
    st.session_state.pop("login_user", None)
    st.session_state["allowed_pages"] = DEFAULT_NORMAL_PAGES[:]
    clear_runtime_caches_after_data_change()
    st.rerun()

current_user = st.session_state.get("login_user", "")
is_admin = bool(st.session_state.get("is_admin", False))
allowed_pages = st.session_state.get("allowed_pages") or get_user_allowed_pages(current_user)
allowed_pages = _normalize_pages(allowed_pages) if not is_admin else ALL_PAGES[:]
st.session_state["allowed_pages"] = allowed_pages

page = st.sidebar.radio(
    label="",
    options=allowed_pages,
    key="nav_radio",
    label_visibility="collapsed",
)

PAGE_SUBTITLES = {
    "经营看板": "老板视角查看院次目标达成、人员负荷、项目结构与预警。",
    "智能排班": "按任务条件推荐团队，并支持确认排班。",
    "批量排班": "批量为任务生成推荐并快速落库。",
    "稽查员管理": "维护稽查员基础信息、负荷与带队能力。",
    "任务管理": "维护项目任务、已定人员与直录排班。",
    "指标统计": "按周/月/季/年统计院次、人员效率与项目结构。",
    "兼职库": "维护不参与自动推荐的兼职人员名单。",
    "城市距离": "维护出发地与目的地距离数据。",
    "城市坐标": "维护城市坐标，供距离自动计算。",
    "模板导入": "批量导入稽查员、任务、城市等基础数据。",
    "日历视图": "查看月度排班日历、排班明细与导出日历。",
    "账号管理": "维护登录账号、密码与板块权限。",
    "数据清理": "清理测试数据与异常记录。",
}

st.title(f"{APP_NAME}｜{page}")
st.caption(PAGE_SUBTITLES.get(page, ""))
st.sidebar.caption(f"当前位置：{page}")

if (not is_admin) and (page not in allowed_pages):
    st.error("当前账号无权限访问该板块，请联系主管理员开通。")
    st.stop()

# -------------------- 经营看板 --------------------
if page == "经营看板":
    st.subheader("经营看板")
    st.caption("老板视角总览院次达成、项目结构、人员负荷与风险预警。")

    today_dt = date.today()
    cy, cm = today_dt.year, today_dt.month
    c1, c2, c3 = st.columns([1, 1, 1])
    board_year = int(c1.number_input("看板年份", min_value=2024, max_value=2035, value=cy, step=1, key="board_year"))
    board_month = int(c2.selectbox("看板月份", list(range(1, 13)), index=max(0, min(11, cm - 1)), key="board_month"))
    board_scope = c3.selectbox("看板口径", ["monthly", "quarterly", "yearly"], format_func=lambda x: {"monthly":"月度总览","quarterly":"季度总览","yearly":"年度总览"}[x], key="board_scope")

    if board_scope == "monthly":
        scope_value = board_month
    elif board_scope == "quarterly":
        scope_value = ((board_month - 1) // 3) + 1
    else:
        scope_value = 0

    start_d, end_d, actual_visits, _, detail_rows = get_progress_stats(board_scope, int(board_year), int(scope_value))
    target_row = get_target_row(board_scope, int(board_year), int(scope_value))
    target_visits = int(target_row.get("target_projects", 0) or 0)
    completion_pct = round(actual_visits / target_visits * 100, 1) if target_visits else 0.0

    cur_month_start, cur_month_end, cur_month_actual, _, _ = get_progress_stats("monthly", int(board_year), int(board_month))
    cur_month_target = int((get_target_row("monthly", int(board_year), int(board_month)) or {}).get("target_projects", 0) or 0)
    current_quarter = ((board_month - 1) // 3) + 1
    q_start, q_end, cur_quarter_actual, _, _ = get_progress_stats("quarterly", int(board_year), int(current_quarter))
    cur_quarter_target = int((get_target_row("quarterly", int(board_year), int(current_quarter)) or {}).get("target_projects", 0) or 0)
    y_start, y_end, cur_year_actual, _, _ = get_progress_stats("yearly", int(board_year), 0)
    cur_year_target = int((get_target_row("yearly", int(board_year), 0) or {}).get("target_projects", 0) or 0)

    with db_session() as db:
        schedules = db.query(Schedule).filter(Schedule.start_date <= end_d, Schedule.end_date >= start_d).all()
        auditors = db.query(Auditor).order_by(Auditor.name.asc()).all()
        tasks = db.query(Task).filter(Task.start_date >= start_d, Task.start_date <= end_d).all()

    unique_task_ids = sorted({int(s.task_id) for s in schedules})
    unique_auditor_ids = sorted({int(s.auditor_id) for s in schedules})
    avg_members = round(len(schedules) / len(unique_task_ids), 1) if unique_task_ids else 0.0
    detail_project_names = sorted({str(r.get("项目名称", "")).strip() for r in detail_rows if str(r.get("项目名称", "")).strip()})

    capacity_map = get_auditor_capacity_map([int(a.id) for a in auditors]) if auditors else {}
    total_days = max(1, (end_d - start_d).days + 1)
    auditor_rows = []
    overloaded_names = []
    idle_risk_names = []
    for a in auditors:
        related = [s for s in schedules if int(s.auditor_id) == int(a.id)]
        day_set = set()
        task_ids = set()
        for srec in related:
            sd = max(start_d, srec.start_date)
            ed = min(end_d, srec.end_date or srec.start_date)
            cur = sd
            while cur <= ed:
                day_set.add(cur)
                cur += timedelta(days=1)
            task_ids.add(int(srec.task_id))
        travel_days = len(day_set)
        idle_days = max(0, total_days - travel_days)
        completed_visits = len(task_ids)
        cap_info = capacity_map.get(int(a.id), {"min_monthly_cases": 4, "max_monthly_cases": 6})
        min_m = int(cap_info.get("min_monthly_cases", 4) or 4)
        max_m = int(cap_info.get("max_monthly_cases", 6) or 6)
        if board_scope == "monthly":
            factor = 1
        elif board_scope == "quarterly":
            factor = 3
        else:
            factor = 12
        std_min = round(min_m * factor, 1)
        std_max = round(max_m * factor, 1)
        std_mid = round((std_min + std_max) / 2, 1) if (std_min + std_max) else 0
        load_pct_num = round(completed_visits / std_mid * 100, 1) if std_mid else 0.0
        overload_pct_num = round(max(0.0, (completed_visits - std_max) / std_max * 100), 1) if std_max else 0.0
        if completed_visits < std_min:
            load_level = "偏低"
            if idle_days >= max(5, total_days // 2):
                idle_risk_names.append(a.name)
        elif completed_visits <= std_max:
            load_level = "正常"
        else:
            load_level = "超负荷"
            overloaded_names.append(a.name)
        auditor_rows.append({
            "稽查员": a.name,
            "已完成院次": completed_visits,
            "出差天数": travel_days,
            "空闲天数": idle_days,
            "标准区间": f"{std_min}-{std_max}",
            "负荷程度": load_level,
            "负荷百分比": load_pct_num,
            "超负荷百分比": overload_pct_num,
        })

    row1 = st.columns(5)
    row1[0].metric("目标院次", target_visits)
    row1[1].metric("已完成院次", actual_visits)
    row1[2].metric("完成率", f"{completion_pct}%")
    row1[3].metric("覆盖项目数", len(unique_task_ids), delta=len(detail_project_names) if detail_project_names else None)
    row1[4].metric("投入稽查员", len(unique_auditor_ids), delta=f"人均{avg_members}" if avg_members else None)

    row2 = st.columns(4)
    month_pct = round(cur_month_actual / cur_month_target * 100, 1) if cur_month_target else 0.0
    quarter_pct = round(cur_quarter_actual / cur_quarter_target * 100, 1) if cur_quarter_target else 0.0
    year_pct = round(cur_year_actual / cur_year_target * 100, 1) if cur_year_target else 0.0
    row2[0].metric(f"{board_year}年{board_month}月达成", f"{cur_month_actual}/{cur_month_target}", delta=f"{month_pct}%")
    row2[1].metric(f"Q{current_quarter}季度达成", f"{cur_quarter_actual}/{cur_quarter_target}", delta=f"{quarter_pct}%")
    row2[2].metric(f"{board_year}年度达成", f"{cur_year_actual}/{cur_year_target}", delta=f"{year_pct}%")
    row2[3].metric("超负荷人数", len(overloaded_names), delta=f"闲置偏高{len(idle_risk_names)}人")

    tab1, tab2, tab3, tab4 = st.tabs(["院次趋势", "人员负荷", "项目结构", "预警清单"])

    with tab1:
        st.markdown("**年度月度趋势**")
        year_rows = []
        for m in range(1, 13):
            _, _, a_visits, _, _ = get_progress_stats("monthly", int(board_year), int(m))
            t_row = get_target_row("monthly", int(board_year), int(m))
            t_visits = int((t_row or {}).get("target_projects", 0) or 0)
            year_rows.append({"月份": f"{m}月", "目标院次": t_visits, "完成院次": a_visits})
        year_df = pd.DataFrame(year_rows)
        st.bar_chart(year_df.set_index("月份")[["目标院次", "完成院次"]], use_container_width=True)

        if board_scope == "monthly":
            trend_rows = get_subperiod_progress_rows("monthly", int(board_year), int(board_month))
            trend_title = f"{board_year}年{board_month}月周度完成"
        elif board_scope == "quarterly":
            trend_rows = get_subperiod_progress_rows("quarterly", int(board_year), int(scope_value))
            trend_title = f"{board_year}年Q{scope_value}月度完成"
        else:
            trend_rows = get_subperiod_progress_rows("yearly", int(board_year), 0)
            trend_title = f"{board_year}年度季度/月度完成"
        st.markdown(f"**{trend_title}**")
        if trend_rows:
            trend_df = pd.DataFrame(trend_rows)
            if "季度" in trend_df.columns:
                trend_df["标签"] = trend_df["标签"] + " (" + trend_df["季度"] + ")"
            st.line_chart(trend_df.set_index("标签")[["目标院次", "完成院次"]], use_container_width=True)
            st.dataframe(trend_df, use_container_width=True, hide_index=True)
        else:
            st.info("当前口径暂无趋势数据")

        st.markdown("**本期项目明细**")
        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
        else:
            st.info("当前口径暂无院次明细")

    with tab2:
        st.markdown("**人员负荷总览**")
        if auditor_rows:
            auditor_df = pd.DataFrame(auditor_rows).sort_values(["超负荷百分比", "已完成院次"], ascending=[False, False])
            cpa, cpb, cpc = st.columns(3)
            cpa.metric("平均出差天数", round(float(auditor_df["出差天数"].mean()), 1))
            cpb.metric("平均空闲天数", round(float(auditor_df["空闲天数"].mean()), 1))
            cpc.metric("平均负荷百分比", f"{round(float(auditor_df['负荷百分比'].mean()), 1)}%")
            st.dataframe(auditor_df, use_container_width=True, hide_index=True)
            st.markdown("**院次/出差天数对比**")
            st.bar_chart(auditor_df.set_index("稽查员")[["已完成院次", "出差天数", "空闲天数"]], use_container_width=True)
            st.markdown("**超负荷百分比 TOP10**")
            top_over_df = auditor_df[["稽查员", "超负荷百分比"]].copy().sort_values("超负荷百分比", ascending=False).head(10)
            st.bar_chart(top_over_df.set_index("稽查员"), use_container_width=True)
        else:
            st.info("当前口径暂无人员负荷数据")

    with tab3:
        st.markdown("**项目结构分析**")
        attr_map = get_task_attribute_map([int(t.id) for t in tasks]) if tasks else {}
        def _dist_frame_dashboard(key_name, label_name):
            counter = {}
            for t in tasks:
                v = (attr_map.get(int(t.id), {}) or {}).get(key_name) or "未填写"
                counter[v] = counter.get(v, 0) + 1
            if not counter:
                return pd.DataFrame(columns=[label_name, "项目数", "占比"])
            total = sum(counter.values()) or 1
            rows = [{label_name: k, "项目数": v, "占比": f"{round(v / total * 100, 1)}%"} for k, v in sorted(counter.items(), key=lambda x: (-x[1], x[0]))]
            return pd.DataFrame(rows)

        xa, xb, xc = st.columns(3)
        with xa:
            st.markdown("**内资/外资结构**")
            dfa = _dist_frame_dashboard("capital_type", "类型")
            st.dataframe(dfa, use_container_width=True, hide_index=True)
            if not dfa.empty:
                st.bar_chart(dfa.set_index("类型")["项目数"], use_container_width=True)
        with xb:
            st.markdown("**分期结构**")
            dfb = _dist_frame_dashboard("project_phase", "分期")
            st.dataframe(dfb, use_container_width=True, hide_index=True)
            if not dfb.empty:
                st.bar_chart(dfb.set_index("分期")["项目数"], use_container_width=True)
        with xc:
            st.markdown("**疾病领域结构**")
            dfc = _dist_frame_dashboard("disease_area", "疾病领域")
            st.dataframe(dfc, use_container_width=True, hide_index=True)
            if not dfc.empty:
                st.bar_chart(dfc.set_index("疾病领域")["项目数"], use_container_width=True)

    with tab4:
        st.markdown("**看板预警**")
        alerts = []
        if target_visits and completion_pct < 80:
            alerts.append({"预警类型": "目标达成", "级别": "高", "说明": f"当前完成率仅 {completion_pct}% ，低于80%。"})
        elif target_visits and completion_pct < 100:
            alerts.append({"预警类型": "目标达成", "级别": "中", "说明": f"当前完成率 {completion_pct}% ，尚未达成目标。"})
        if overloaded_names:
            alerts.append({"预警类型": "人员负荷", "级别": "高", "说明": f"超负荷稽查员：{', '.join(overloaded_names[:8])}{'...' if len(overloaded_names) > 8 else ''}"})
        if idle_risk_names:
            alerts.append({"预警类型": "人员闲置", "级别": "中", "说明": f"空闲较高稽查员：{', '.join(idle_risk_names[:8])}{'...' if len(idle_risk_names) > 8 else ''}"})
        attr_map = get_task_attribute_map([int(t.id) for t in tasks]) if tasks else {}
        missing_attr_tasks = []
        for t in tasks:
            extra = attr_map.get(int(t.id), {}) or {}
            miss = []
            if not (extra.get("capital_type") or "").strip():
                miss.append("内外资")
            if not (extra.get("project_phase") or "").strip():
                miss.append("分期")
            if not (extra.get("disease_area") or "").strip():
                miss.append("疾病领域")
            if miss:
                missing_attr_tasks.append(f"{t.project_name}（缺少：{'/'.join(miss)}）")
        if missing_attr_tasks:
            alerts.append({"预警类型": "项目信息", "级别": "中", "说明": "；".join(missing_attr_tasks[:8]) + ("..." if len(missing_attr_tasks) > 8 else "")})
        no_team_task_names = []
        for t in tasks:
            if int(t.id) not in unique_task_ids:
                no_team_task_names.append(str(t.project_name))
        if no_team_task_names:
            alerts.append({"预警类型": "排班覆盖", "级别": "中", "说明": f"当前口径内未见排班记录项目：{', '.join(no_team_task_names[:8])}{'...' if len(no_team_task_names) > 8 else ''}"})
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
        else:
            st.success("当前看板未发现明显风险预警。")

# -------------------- 智能排班 --------------------
if page == "智能排班":
    st.subheader("智能排班")
    st.caption("先按硬约束筛选，再按距离优先 + 适度负荷均衡评分推荐。")

    data_version = int(st.session_state.get("data_version", 0))
    task_rows = get_tasks_for_ui(data_version)
    schedules_recent_rows = get_recent_schedule_rows(data_version, 120)

    if not task_rows:
        st.info("请先在【任务管理】中录入任务。")
    else:
        task_options = {row["label"]: row["id"] for row in task_rows}
        selected_label = st.selectbox("选择任务", list(task_options.keys()), key="smart_task_select")
        selected_task_id = task_options[selected_label]

        col_a, _ = st.columns([1, 3])
        if col_a.button("生成推荐", type="primary", key="gen_reco_btn"):
            st.session_state["recommend_result"] = get_recommendation_payload(selected_task_id, data_version)
            st.rerun()

        rec = st.session_state.get("recommend_result")
        if rec and rec.get("task_id") == selected_task_id:
            with db_session() as db:
                task = db.query(Task).filter(Task.id == selected_task_id).first()
            if task:
                st.info(
                    f"已选择：{task.project_name}（{task.site_city}，{d2s(task.start_date)}，{task.required_days}天，{task.required_headcount}人；需要A带队：{'是' if task.need_expert else '否'}）"
                )
            if rec.get("error"):
                st.error(rec["error"])
            team = _dict_to_team(rec.get("team"))
            if team:
                st.subheader("系统推荐团队方案")
                st.write(
                    f"**负责人：** {team.leader.auditor_name}（{team.leader.group_level}，{'可带队' if team.leader.can_lead_team else '不可带队'}，出发地 {team.leader.from_city}，{team.leader.km:.0f}km，评分 {team.leader.score}）"
                )
                if team.members:
                    st.write(
                        "**组员：** "
                        + "； ".join(
                            [
                                f"{m.auditor_name}（{m.group_level}，{m.from_city}，{m.km:.0f}km，评分 {m.score}）"
                                for m in team.members
                            ]
                        )
                    )
                else:
                    st.write("**组员：** 无")
                st.caption(f"{team.notes}｜团队评分 {team.team_score}")
                default_member_ids = ",".join([str(m.auditor_id) for m in team.members])
                member_ids_text = st.text_input("确认指派前，可手工调整组员ID（逗号分隔）", value=default_member_ids, key="member_ids_text")
                if st.button("确认指派", type="primary", key="confirm_assign_btn"):
                    ids = [x for x in re.split(r"[，,\s]+", member_ids_text.strip()) if x.strip()]
                    member_ids = []
                    for x in ids:
                        try:
                            member_ids.append(int(x))
                        except Exception:
                            pass
                    with db_session() as db:
                        task = db.query(Task).filter(Task.id == selected_task_id).first()
                        ok, msg = assign_team_to_task(db, task, int(team.leader.auditor_id), member_ids)
                        if not ok:
                            db.rollback()
                            st.error(msg)
                            st.stop()
                        if not safe_commit(db, context=f"确认指派：task#{selected_task_id}"):
                            st.stop()
                    clear_runtime_caches_after_data_change()
                    st.success("已确认指派")
                    st.rerun()

            cands = rec.get("candidates") or []
            if cands:
                st.subheader("候选人 TOP25")
                rows = []
                for i, d in enumerate(cands, start=1):
                    c = _dict_to_candidate(d)
                    rows.append(
                        {
                            "排名": i,
                            "姓名": c.auditor_name,
                            "组别": c.group_level,
                            "带队": "是" if c.can_lead_team else "否",
                            "出发地": c.from_city,
                            "距离(km)": round(float(c.km), 1),
                            "评分": c.score,
                            "解释": c.explain,
                        }
                    )
                show_table(rows, 420)

    st.subheader("最近排班记录（TOP120）")
    current_schedule_page_rows = show_paginated_table(schedules_recent_rows, "recent_schedule", 360, default_page_size=20)
    if schedules_recent_rows:
        delete_sid = st.selectbox("删除排班记录（按ID）", [r["ID"] for r in (current_schedule_page_rows or schedules_recent_rows)], key="delete_schedule_select")
        if st.button("删除所选排班记录", key="delete_schedule_btn"):
            with db_session() as db:
                obj = db.query(Schedule).filter(Schedule.id == delete_sid).first()
                if obj:
                    db.delete(obj)
                    if not safe_commit(db, context=f"删除排班记录：schedule#{delete_sid}"):
                        st.stop()
            clear_runtime_caches_after_data_change()
            st.success("已删除")
            st.rerun()

# -------------------- 批量排班 --------------------
elif page == "批量排班":
    st.subheader("批量排班")
    st.caption("只会处理未排过的任务；按 need_expert 优先 > 人数多优先 > 开始日期早 排序。")

    c1, c2, c3 = st.columns([1, 1, 1])
    date_start = c1.date_input("开始日期", value=date.today(), key="batch_start")
    date_end = c2.date_input("结束日期", value=date.today() + timedelta(days=30), key="batch_end")
    mode = c3.selectbox(
        "模式",
        ["greedy", "optimized"],
        format_func=lambda x: "快速模式（优先效率）" if x == "greedy" else "优化模式（优先成本与均衡）",
        key="batch_mode",
    )
    if st.button("开始批量排班", type="primary", key="batch_run_btn"):
        with db_session() as db:
            report = run_batch_schedule(db, date_start, date_end, mode)
        st.session_state["batch_report"] = report
        st.rerun()

    report = st.session_state.get("batch_report")
    if report:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"已自动排班（{len(report.get('assigned', []))}）")
            if report.get("assigned"):
                for a in report["assigned"]:
                    st.write(f"**#{a['task_id']} {a['project']}**")
                    st.caption(f"负责人：{a['leader']}；组员：{', '.join(a['members']) if a['members'] else '无'}")
            else:
                st.info("无")
        with c2:
            st.subheader(f"跳过任务（{len(report.get('skipped', []))}）")
            if report.get("skipped"):
                for s in report["skipped"]:
                    st.write(f"**#{s['task_id']} {s['project']}**")
                    st.caption(f"原因：{s['reason']}")
            else:
                st.info("无")

# -------------------- 稽查员管理 --------------------
elif page == "稽查员管理":
    st.subheader("稽查员管理")

    with st.form("auditor_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("姓名*")
        gender = c2.selectbox("性别*", ["女", "男"], index=0)
        group_level = c3.selectbox("等级*", ["A", "B", "C"], index=1)
        can_lead = c4.selectbox("可带队*", ["是", "否"], index=0)

        c5, c6, c7, c8 = st.columns(4)
        base_city = c5.text_input("常驻城市*")
        max_weekly_tasks = c6.number_input("每周上限", min_value=0, value=1, step=1)
        status_cn = c7.selectbox("状态", ["在岗", "请假", "冻结"])
        monthly_cases = c8.number_input("本月已排院次", min_value=0, value=0, step=1)

        c9, c10, c11, c12 = st.columns(4)
        travel_days = c9.number_input("本月差旅天数", min_value=0, value=0, step=1)
        continuous_days = c10.number_input("连续工作天数", min_value=0, value=0, step=1)
        last_city = c11.text_input("上次结束城市（可空）")
        last_date = c12.date_input("上次结束日期*", value=date.today())

        if st.form_submit_button("新增稽查员", type="primary"):
            if not name.strip() or not base_city.strip():
                st.error("姓名、常驻城市必填。")
            else:
                with db_session() as db:
                    db.add(
                        Auditor(
                            name=name.strip(),
                            gender=gender,
                            group_level=group_level,
                            can_lead_team=(can_lead == "是"),
                            base_city=base_city.strip(),
                            max_weekly_tasks=int(max_weekly_tasks),
                            status=STATUS_MAP[status_cn],
                            monthly_cases=int(monthly_cases),
                            travel_days=int(travel_days),
                            continuous_days=int(continuous_days),
                            last_task_end_city=last_city.strip() or None,
                            last_task_end_date=last_date,
                        )
                    )
                    if not safe_commit(db, context=f"新增稽查员：{name.strip()}"):
                        st.stop()
                clear_runtime_caches_after_data_change()
                st.success("已新增")
                st.rerun()

    with db_session() as db:
        auditors = db.query(Auditor).order_by(Auditor.id.desc()).all()

    rows = []
    for a in auditors:
        rows.append(
            {
                "ID": a.id,
                "姓名": a.name,
                "性别": a.gender,
                "等级": a.group_level,
                "可带队": "是" if a.can_lead_team else "否",
                "常驻城市": a.base_city,
                "周上限": a.max_weekly_tasks,
                "状态": STATUS_MAP_REV.get(a.status, a.status),
                "本月院次": a.monthly_cases,
                "差旅天数": a.travel_days,
                "连续天数": a.continuous_days,
                "上次结束城市": a.last_task_end_city or "",
                "上次结束日期": d2s(a.last_task_end_date),
            }
        )

    current_auditor_page_rows = show_paginated_table(rows, "auditor_list", 320, default_page_size=20)

    if auditors:
        auditor_options = {
            f"#{a.id} {a.name}｜{a.base_city}｜{a.group_level}": a.id
            for a in auditors
        }
        selected_auditor_label = st.selectbox("选择要编辑的稽查员", list(auditor_options.keys()), key="edit_auditor_select")
        selected_auditor_id = auditor_options[selected_auditor_label]
        selected_auditor = next((a for a in auditors if a.id == selected_auditor_id), None)

        if selected_auditor:
            st.divider()
            st.subheader(f"编辑稽查员 #{selected_auditor.id}")

            with st.form("edit_auditor_form", clear_on_submit=False):
                c1, c2, c3, c4 = st.columns(4)
                edit_name = c1.text_input("姓名*", value=selected_auditor.name or "")
                edit_gender = c2.selectbox("性别*", ["女", "男"], index=0 if (selected_auditor.gender or "女") == "女" else 1)
                edit_group = c3.selectbox("等级*", ["A", "B", "C"], index=["A", "B", "C"].index(selected_auditor.group_level or "B"))
                edit_can_lead = c4.selectbox("可带队*", ["是", "否"], index=0 if selected_auditor.can_lead_team else 1)

                c5, c6, c7, c8 = st.columns(4)
                edit_base_city = c5.text_input("常驻城市*", value=selected_auditor.base_city or "")
                edit_max_weekly_tasks = c6.number_input("每周上限", min_value=0, value=int(selected_auditor.max_weekly_tasks or 0), step=1)
                edit_status = c7.selectbox(
                    "状态",
                    ["在岗", "请假", "冻结"],
                    index=["在岗", "请假", "冻结"].index(STATUS_MAP_REV.get(selected_auditor.status, "在岗")),
                )
                edit_monthly_cases = c8.number_input("本月已排院次", min_value=0, value=int(selected_auditor.monthly_cases or 0), step=1)

                c9, c10, c11, c12 = st.columns(4)
                edit_travel_days = c9.number_input("本月差旅天数", min_value=0, value=int(selected_auditor.travel_days or 0), step=1)
                edit_continuous_days = c10.number_input("连续工作天数", min_value=0, value=int(selected_auditor.continuous_days or 0), step=1)
                edit_last_city = c11.text_input("上次结束城市（可空）", value=selected_auditor.last_task_end_city or "")
                edit_last_date = c12.date_input("上次结束日期*", value=selected_auditor.last_task_end_date or date.today())

                b1, b2 = st.columns(2)
                save_edit = b1.form_submit_button("保存当前稽查员修改", type="primary")
                delete_edit = b2.form_submit_button("删除当前稽查员")

            if save_edit:
                ok = update_auditor_record(
                    auditor_id=selected_auditor.id,
                    name=edit_name,
                    gender=edit_gender,
                    group_level=edit_group,
                    can_lead_team=(edit_can_lead == "是"),
                    base_city=edit_base_city,
                    max_weekly_tasks=int(edit_max_weekly_tasks),
                    status_cn=edit_status,
                    monthly_cases=int(edit_monthly_cases),
                    travel_days=int(edit_travel_days),
                    continuous_days=int(edit_continuous_days),
                    last_task_end_city=edit_last_city,
                    last_task_end_date=edit_last_date,
                )
                if ok:
                    clear_runtime_caches_after_data_change()
                    st.success("稽查员修改已保存")
                    st.rerun()

            if delete_edit:
                ok = delete_auditor_record(selected_auditor.id)
                if ok:
                    clear_runtime_caches_after_data_change()
                    st.success("稽查员已删除")
                    st.rerun()

# -------------------- 任务管理 --------------------
elif page == "任务管理":
    st.subheader("任务管理")

    with st.form("task_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        project_name = c1.text_input("项目名称*")
        customer_name = c2.text_input("客户/申办方（可空）")
        need_expert = c3.selectbox("需要A带队", ["否", "是"])

        c4, c5, c6, c7 = st.columns(4)
        required_headcount = c4.number_input("所需人数", min_value=1, value=1, step=1)
        required_days = c5.number_input("任务天数", min_value=1, value=1, step=1)
        required_gender = c6.selectbox("性别要求", ["不限", "男", "女"])
        site_city = c7.text_input("中心城市*")

        c8, c9, c10 = st.columns(3)
        specified = c8.text_input("硬指定人员（可空，支持 ，、分隔）")
        preferred = c9.text_input("软指定专家/老师（可空）")
        start_date = c10.date_input("开始日期*", value=date.today())
        c11, c12, c13 = st.columns(3)
        capital_type = c11.selectbox("项目属性", ["", "内资", "外资"])
        project_phase_pick = c12.selectbox("项目分期", [""] + PRESET_PROJECT_PHASES + ["其他（手填）"], index=0)
        disease_area_pick = c13.selectbox("疾病领域", [""] + PRESET_DISEASE_AREAS + ["其他（手填）"], index=0)
        c14, c15 = st.columns(2)
        project_phase_other = c14.text_input("其他分期（可空）") if project_phase_pick == "其他（手填）" else ""
        disease_area_other = c15.text_input("其他疾病领域（可空）") if disease_area_pick == "其他（手填）" else ""
        project_phase = _merge_preset_and_other(project_phase_pick, project_phase_other)
        disease_area = _merge_preset_and_other(disease_area_pick, disease_area_other)
        default_end = start_date + timedelta(days=max(1, int(required_days)) - 1)
        end_date = st.date_input("结束日期*", value=default_end)

        if st.form_submit_button("新增任务", type="primary"):
            if not project_name.strip() or not site_city.strip():
                st.error("项目名称、中心城市必填。")
            elif end_date < start_date:
                st.error("结束日期不能早于开始日期。")
            else:
                with db_session() as db:
                    obj = Task(
                        project_name=project_name.strip(),
                        customer_name=customer_name.strip() or None,
                        need_expert=(need_expert == "是"),
                        required_headcount=int(required_headcount),
                        required_days=int(required_days),
                        required_gender=required_gender,
                        specified_auditors=specified.strip() or None,
                        preferred_experts=preferred.strip() or None,
                        site_city=site_city.strip(),
                        start_date=start_date,
                        end_date=end_date,
                    )
                    db.add(obj)
                    if not safe_commit(db, context=f"新增任务：{project_name.strip()}"):
                        st.stop()
                    save_task_attributes(int(obj.id), capital_type=capital_type, project_phase=project_phase, disease_area=disease_area)
                clear_runtime_caches_after_data_change()
                st.success("已新增")
                st.rerun()

    with db_session() as db:
        tasks = db.query(Task).order_by(Task.id.desc()).all()
        auditors = db.query(Auditor).order_by(Auditor.name.asc()).all()

    attr_map = get_task_attribute_map([int(t.id) for t in tasks]) if tasks else {}
    rows = []
    for t in tasks:
        extra = attr_map.get(int(t.id), {})
        rows.append(
            {
                "ID": t.id,
                "项目": t.project_name,
                "客户": t.customer_name or "",
                "属性": extra.get("capital_type") or "",
                "分期": extra.get("project_phase") or "",
                "疾病领域": extra.get("disease_area") or "",
                "需要A": "是" if t.need_expert else "否",
                "人数": t.required_headcount,
                "天数": t.required_days,
                "性别": t.required_gender,
                "硬指定": t.specified_auditors or "",
                "软指定": t.preferred_experts or "",
                "城市": t.site_city,
                "开始": d2s(t.start_date),
                "结束": d2s(t.end_date),
            }
        )

    current_task_page_rows = show_paginated_table(rows, "task_list", 320, default_page_size=20)

    if tasks:
        task_options = {
            f"#{t.id} {t.project_name}｜{t.site_city}｜{d2s(t.start_date)}": t.id
            for t in tasks
        }
        selected_task_label = st.selectbox("选择要编辑的任务", list(task_options.keys()), key="edit_task_select")
        selected_task_id = task_options[selected_task_label]
        selected_task = next((t for t in tasks if t.id == selected_task_id), None)

        if selected_task:
            selected_task_attrs = get_task_attributes(int(selected_task.id))
            st.divider()
            st.subheader(f"编辑任务 #{selected_task.id}")

            with st.form("edit_task_form", clear_on_submit=False):
                c1, c2, c3 = st.columns(3)
                edit_project_name = c1.text_input("项目名称*", value=selected_task.project_name or "")
                edit_customer_name = c2.text_input("客户/申办方（可空）", value=selected_task.customer_name or "")
                edit_need_expert = c3.selectbox("需要A带队", ["否", "是"], index=1 if selected_task.need_expert else 0)

                c4, c5, c6, c7 = st.columns(4)
                edit_required_headcount = c4.number_input("所需人数", min_value=1, value=int(selected_task.required_headcount or 1), step=1)
                edit_required_days = c5.number_input("任务天数", min_value=1, value=int(selected_task.required_days or 1), step=1)
                edit_required_gender = c6.selectbox(
                    "性别要求",
                    ["不限", "男", "女"],
                    index=["不限", "男", "女"].index(selected_task.required_gender or "不限"),
                )
                edit_site_city = c7.text_input("中心城市*", value=selected_task.site_city or "")

                c8, c9, c10 = st.columns(3)
                edit_specified = c8.text_input("硬指定人员（可空）", value=selected_task.specified_auditors or "")
                edit_preferred = c9.text_input("软指定专家/老师（可空）", value=selected_task.preferred_experts or "")
                edit_start_date = c10.date_input("开始日期*", value=selected_task.start_date or date.today())
                c11, c12, c13 = st.columns(3)
                edit_capital_type = c11.selectbox("项目属性", ["", "内资", "外资"], index=["", "内资", "外资"].index((selected_task_attrs.get("capital_type") or "") if (selected_task_attrs.get("capital_type") or "") in ["", "内资", "外资"] else ""))
                _phase_pick, _phase_other = _preset_or_other(selected_task_attrs.get("project_phase") or "", PRESET_PROJECT_PHASES)
                _disease_pick, _disease_other = _preset_or_other(selected_task_attrs.get("disease_area") or "", PRESET_DISEASE_AREAS)
                edit_project_phase_pick = c12.selectbox("项目分期", [""] + PRESET_PROJECT_PHASES + ["其他（手填）"], index=([""] + PRESET_PROJECT_PHASES + ["其他（手填）"]).index(_phase_pick if _phase_pick in ([""] + PRESET_PROJECT_PHASES + ["其他（手填）"]) else ""))
                edit_disease_area_pick = c13.selectbox("疾病领域", [""] + PRESET_DISEASE_AREAS + ["其他（手填）"], index=([""] + PRESET_DISEASE_AREAS + ["其他（手填）"]).index(_disease_pick if _disease_pick in ([""] + PRESET_DISEASE_AREAS + ["其他（手填）"]) else ""))
                c14, c15 = st.columns(2)
                edit_project_phase_other = c14.text_input("其他分期（可空）", value=_phase_other) if edit_project_phase_pick == "其他（手填）" else ""
                edit_disease_area_other = c15.text_input("其他疾病领域（可空）", value=_disease_other) if edit_disease_area_pick == "其他（手填）" else ""
                edit_project_phase = _merge_preset_and_other(edit_project_phase_pick, edit_project_phase_other)
                edit_disease_area = _merge_preset_and_other(edit_disease_area_pick, edit_disease_area_other)
                edit_end_date = st.date_input("结束日期*", value=selected_task.end_date or edit_start_date)

                b1, b2 = st.columns(2)
                save_task = b1.form_submit_button("保存当前任务修改", type="primary")
                delete_task = b2.form_submit_button("删除当前任务")

            if save_task:
                ok = update_task_record(
                    task_id=selected_task.id,
                    project_name=edit_project_name,
                    customer_name=edit_customer_name,
                    need_expert=(edit_need_expert == "是"),
                    required_headcount=int(edit_required_headcount),
                    required_days=int(edit_required_days),
                    required_gender=edit_required_gender,
                    specified_auditors=edit_specified,
                    preferred_experts=edit_preferred,
                    site_city=edit_site_city,
                    start_date_value=edit_start_date,
                    end_date_value=edit_end_date,
                    capital_type=edit_capital_type,
                    project_phase=edit_project_phase,
                    disease_area=edit_disease_area,
                )
                if ok:
                    clear_runtime_caches_after_data_change()
                    st.success("任务修改已保存")
                    st.rerun()

            if delete_task:
                ok = delete_task_record(selected_task.id)
                if ok:
                    clear_runtime_caches_after_data_change()
                    st.success("任务已删除")
                    st.rerun()

            st.divider()
            st.subheader("已定项目人员录入 / 直录排班")
            st.caption("适用于项目经理已完成排班，不需要系统推荐。支持内部稽查员与兼职人员，且可分别设置每个人在项目中的起止日期。")

            part_time_rows = get_part_time_staff_rows(active_only=True)
            auditor_name_to_id = {a.name: a.id for a in auditors}
            auditor_name_options = [a.name for a in auditors]
            part_time_options = [r["name"] for r in part_time_rows]

            existing_direct = get_direct_assignments(int(selected_task.id))
            if existing_direct:
                direct_df = pd.DataFrame([
                    {
                        "项目名称": r.get("project_name", "") or (selected_task.project_name or ""),
                        "类型": "兼职" if bool(r.get("is_part_time")) else "内部稽查员",
                        "人员姓名": r.get("person_name", ""),
                        "角色": "组长" if str(r.get("role", "")) == "leader" else "成员",
                        "开始日期": str(r.get("start_date", "")),
                        "结束日期": str(r.get("end_date", "")),
                        "备注": r.get("notes", "") or "",
                    }
                    for r in existing_direct
                ])
            else:
                default_names = parse_name_list(selected_task.specified_auditors or "")
                if default_names:
                    direct_df = pd.DataFrame([
                        {
                            "项目名称": selected_task.project_name or "",
                            "类型": "内部稽查员" if nm in auditor_name_to_id else "兼职",
                            "人员姓名": nm,
                            "角色": "成员",
                            "开始日期": d2s(selected_task.start_date),
                            "结束日期": d2s(selected_task.end_date),
                            "备注": "",
                        }
                        for nm in default_names
                    ])
                else:
                    direct_df = pd.DataFrame(columns=["项目名称", "类型", "人员姓名", "角色", "开始日期", "结束日期", "备注"])

            with st.form("direct_assign_form", clear_on_submit=False):
                edited_direct = st.data_editor(
                    direct_df,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="dynamic",
                    key=f"direct_assign_editor_{selected_task.id}",
                    column_config={
                        "项目名称": st.column_config.TextColumn(help="支持查看或补充项目名称"),
                        "类型": st.column_config.SelectboxColumn(options=["内部稽查员", "兼职"]),
                        "人员姓名": st.column_config.TextColumn(help="内部稽查员可填写系统内姓名；兼职可填写兼职库姓名"),
                        "角色": st.column_config.SelectboxColumn(options=["组长", "成员"]),
                        "开始日期": st.column_config.TextColumn(help="格式 YYYY-MM-DD"),
                        "结束日期": st.column_config.TextColumn(help="格式 YYYY-MM-DD"),
                        "备注": st.column_config.TextColumn(),
                    },
                )
                c1, c2 = st.columns(2)
                save_direct = c1.form_submit_button("保存已定项目人员")
                sync_direct = c2.form_submit_button("按已定人员直接录入排班", type="primary")

            def _normalize_direct_rows(df_in):
                rows_out = []
                for _, r in pd.DataFrame(df_in).iterrows():
                    person_name = str(r.get("人员姓名", "")).strip()
                    if not person_name:
                        continue
                    role = "leader" if str(r.get("角色", "")).strip() == "组长" else "member"
                    is_part_time = str(r.get("类型", "")).strip() == "兼职"
                    sd = safe_parse_date(r.get("开始日期"))
                    ed = safe_parse_date(r.get("结束日期"))
                    if not sd or not ed:
                        continue
                    rows_out.append(
                        {
                            "auditor_id": None if is_part_time else auditor_name_to_id.get(person_name),
                            "person_name": person_name,
                            "is_part_time": is_part_time,
                            "role": role,
                            "start_date": sd,
                            "end_date": ed,
                            "project_name": normalize_text(r.get("项目名称")) or normalize_text(selected_task.project_name),
                            "notes": str(r.get("备注", "")).strip(),
                        }
                    )
                return rows_out

            if save_direct:
                rows_to_save = _normalize_direct_rows(edited_direct)
                replace_direct_assignments(int(selected_task.id), rows_to_save)
                st.success("已定项目人员已保存")
                st.rerun()

            if sync_direct:
                rows_to_save = _normalize_direct_rows(edited_direct)
                if not rows_to_save:
                    st.error("请先录入至少1条已定项目人员")
                else:
                    replace_direct_assignments(int(selected_task.id), rows_to_save)
                    ok, msg = sync_task_schedules_from_direct_assignments(selected_task)
                    if ok:
                        clear_runtime_caches_after_data_change()
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


# -------------------- 页面：指标统计 --------------------
elif page == "指标统计":
    st.subheader("指标统计")
    st.caption("按周、月、季、年录入目标院次，自动统计完成率、趋势图、稽查员出差/空闲/负荷情况，以及项目属性分布。")

    c1, c2, c3 = st.columns(3)
    period_type = c1.selectbox("统计周期", ["weekly", "monthly", "quarterly", "yearly"], format_func=lambda x: {"weekly":"周度","monthly":"月度","quarterly":"季度","yearly":"年度"}[x])
    year = c2.number_input("年份", min_value=2024, max_value=2035, value=date.today().year, step=1)
    if period_type == "weekly":
        period_value = c3.selectbox("周次", list(range(1, 54)), index=max(0, min(52, date.today().isocalendar().week - 1)))
    elif period_type == "monthly":
        period_value = c3.selectbox("月份", list(range(1, 13)), index=max(0, date.today().month - 1))
    elif period_type == "quarterly":
        period_value = c3.selectbox("季度", [1, 2, 3, 4], index=(date.today().month - 1)//3)
    else:
        period_value = 0
        c3.markdown("**全年**")

    target = get_target_row(period_type, int(year), int(period_value))
    with st.form("target_form", clear_on_submit=False):
        target_projects = st.number_input("目标院次数量", min_value=0, value=int(target.get("target_projects", 0) or 0), step=1)
        if st.form_submit_button("保存院次指标", type="primary"):
            ok = save_target_row(period_type, int(year), int(period_value), int(target_projects), 0)
            if ok:
                st.success("院次指标已保存")
                st.rerun()
            else:
                st.error("院次指标保存失败，请重试")

    start_d, end_d, actual_visits, _, detail_rows = get_progress_stats(period_type, int(year), int(period_value))
    target = get_target_row(period_type, int(year), int(period_value))
    t_visits = int(target.get("target_projects", 0) or 0)
    completion_pct = round(actual_visits / t_visits * 100, 1) if t_visits else 0.0

    summary_df = pd.DataFrame([
        {"指标": "目标院次", "数值": t_visits},
        {"指标": "已完成院次", "数值": actual_visits},
        {"指标": "完成率", "数值": completion_pct},
    ])
    st.write(f"统计区间：{d2s(start_d)} ~ {d2s(end_d)}")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    colx, coly, colz = st.columns(3)
    colx.metric("目标院次", t_visits)
    coly.metric("已完成院次", actual_visits)
    colz.metric("完成率", f"{completion_pct}%")

    sub_rows = get_subperiod_progress_rows(period_type, int(year), int(period_value))
    if sub_rows:
        st.subheader("完成趋势图")
        chart_df = pd.DataFrame(sub_rows)
        if period_type == "yearly" and "季度" in chart_df.columns:
            chart_df["标签"] = chart_df["标签"] + " (" + chart_df["季度"] + ")"
        st.bar_chart(chart_df.set_index("标签")[["目标院次", "完成院次"]], use_container_width=True)
        st.dataframe(chart_df, use_container_width=True, hide_index=True)

    st.subheader("院次完成明细")
    if detail_rows:
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    else:
        st.info("该统计区间暂无项目数据")

    with db_session() as db:
        schedules = db.query(Schedule).filter(Schedule.start_date <= end_d, Schedule.end_date >= start_d).all()
        auditors = db.query(Auditor).order_by(Auditor.name.asc()).all()
        tasks = db.query(Task).filter(Task.start_date >= start_d, Task.start_date <= end_d).all()

    capacity_map = get_auditor_capacity_map([int(a.id) for a in auditors]) if auditors else {}
    st.subheader("稽查员月度院次标准设置")
    cap_df = pd.DataFrame([
        {
            "稽查员": a.name,
            "最小标准": int((capacity_map.get(int(a.id), {}) or {}).get("min_monthly_cases", 4) or 4),
            "最大标准": int((capacity_map.get(int(a.id), {}) or {}).get("max_monthly_cases", 6) or 6),
        }
        for a in auditors
    ])
    if not cap_df.empty:
        edited_cap = st.data_editor(
            cap_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="auditor_capacity_editor",
            column_config={
                "最小标准": st.column_config.NumberColumn(min_value=0, step=1),
                "最大标准": st.column_config.NumberColumn(min_value=0, step=1),
            },
        )
        if st.button("保存稽查员月度院次标准", type="primary"):
            name_to_id = {a.name: int(a.id) for a in auditors}
            for _, r in pd.DataFrame(edited_cap).iterrows():
                aid = name_to_id.get(str(r.get("稽查员", "")).strip())
                if aid:
                    save_auditor_capacity_target(aid, _safe_int(r.get("最小标准"), 4), _safe_int(r.get("最大标准"), 6))
            clear_runtime_caches_after_data_change()
            st.success("稽查员月度院次标准已保存")
            st.rerun()

    auditor_rows = []
    total_days = max(1, (end_d - start_d).days + 1)
    for a in auditors:
        related = [s for s in schedules if int(s.auditor_id) == int(a.id)]
        day_set = set()
        task_ids = set()
        for srec in related:
            sd = max(start_d, srec.start_date)
            ed = min(end_d, srec.end_date or srec.start_date)
            cur = sd
            while cur <= ed:
                day_set.add(cur)
                cur += timedelta(days=1)
            task_ids.add(int(srec.task_id))
        travel_days = len(day_set)
        idle_days = max(0, total_days - travel_days)
        completed_visits = len(task_ids)
        cap_info = capacity_map.get(int(a.id), {"min_monthly_cases": 4, "max_monthly_cases": 6})
        min_m = int(cap_info.get("min_monthly_cases", 4) or 4)
        max_m = int(cap_info.get("max_monthly_cases", 6) or 6)
        if period_type == "weekly":
            factor = 0.25
        elif period_type == "monthly":
            factor = 1
        elif period_type == "quarterly":
            factor = 3
        else:
            factor = 12
        std_min = round(min_m * factor, 1)
        std_max = round(max_m * factor, 1)
        std_mid = round((std_min + std_max) / 2, 1)
        load_pct = round(completed_visits / std_mid * 100, 1) if std_mid else 0.0
        overload_pct = round(max(0.0, (completed_visits - std_max) / std_max * 100), 1) if std_max else 0.0
        if completed_visits < std_min:
            load_level = "偏低"
        elif completed_visits <= std_max:
            load_level = "正常"
        else:
            load_level = "超负荷"
        auditor_rows.append({
            "稽查员": a.name,
            "已完成院次": completed_visits,
            "出差天数": travel_days,
            "空闲天数": idle_days,
            "月度标准": f"{min_m}-{max_m}",
            "折算标准": f"{std_min}-{std_max}",
            "负荷程度": load_level,
            "负荷百分比": f"{load_pct}%",
            "超负荷百分比": f"{overload_pct}%",
        })

    st.subheader("稽查员效率统计")
    if auditor_rows:
        auditor_df = pd.DataFrame(auditor_rows)
        st.dataframe(auditor_df, use_container_width=True, hide_index=True)
        plot_df = auditor_df[["稽查员", "已完成院次", "出差天数", "空闲天数"]].set_index("稽查员")
        st.bar_chart(plot_df, use_container_width=True)
    else:
        st.info("暂无稽查员统计数据")

    attr_map = get_task_attribute_map([int(t.id) for t in tasks]) if tasks else {}
    def _dist_frame(key_name):
        counter = {}
        for t in tasks:
            v = (attr_map.get(int(t.id), {}) or {}).get(key_name) or "未填写"
            counter[v] = counter.get(v, 0) + 1
        if not counter:
            return pd.DataFrame(columns=[key_name, "院次", "占比"])
        total = sum(counter.values()) or 1
        rows = [{key_name: k, "院次": v, "占比": f"{round(v / total * 100, 1)}%"} for k, v in sorted(counter.items(), key=lambda x: (-x[1], x[0]))]
        return pd.DataFrame(rows)

    st.subheader("项目结构实时分析")
    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("**内资/外资占比**")
        dfa = _dist_frame("capital_type")
        st.dataframe(dfa, use_container_width=True, hide_index=True)
        if not dfa.empty:
            st.bar_chart(dfa.set_index("capital_type")["院次"], use_container_width=True)
    with cb:
        st.markdown("**项目分期占比**")
        dfb = _dist_frame("project_phase")
        st.dataframe(dfb, use_container_width=True, hide_index=True)
        if not dfb.empty:
            st.bar_chart(dfb.set_index("project_phase")["院次"], use_container_width=True)
    with cc:
        st.markdown("**疾病领域占比**")
        dfc = _dist_frame("disease_area")
        st.dataframe(dfc, use_container_width=True, hide_index=True)
        if not dfc.empty:
            st.bar_chart(dfc.set_index("disease_area")["院次"], use_container_width=True)

# -------------------- 页面：兼职库 --------------------
elif page == "兼职库":
    st.subheader("兼职库")
    st.caption("兼职人员不会进入系统自动推荐，仅用于已定项目人员录入与直录排班场景。")

    with st.form("part_time_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        pt_name = c1.text_input("兼职姓名*")
        pt_city = c2.text_input("常驻城市（可空）")
        pt_active = c3.selectbox("状态", ["启用", "停用"], index=0)
        pt_note = st.text_input("备注（可空）")
        if st.form_submit_button("保存兼职人员", type="primary"):
            ok, msg = save_part_time_staff(pt_name, pt_city, pt_note, is_active=(pt_active == "启用"))
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    rows = get_part_time_staff_rows(active_only=False)
    if rows:
        show_table([
            {
                "ID": r["id"],
                "姓名": r["name"],
                "常驻城市": r.get("base_city") or "",
                "状态": "启用" if int(r.get("is_active", 0)) == 1 else "停用",
                "备注": r.get("note") or "",
            } for r in rows
        ], 280)

        deletable = {f"#{r['id']} {r['name']}": r["id"] for r in rows}
        c1, c2 = st.columns([2, 1])
        del_label = c1.selectbox("选择要删除的兼职人员", list(deletable.keys()))
        if c2.button("删除兼职人员"):
            ok, msg = delete_part_time_staff(deletable[del_label])
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    else:
        st.info("暂无兼职人员")
# -------------------- 城市距离 --------------------
elif page == "城市距离":
    st.subheader("城市距离")
    st.caption("系统会优先读取距离表；若未命中，会尝试按城市坐标自动计算并写回缓存。")
    with st.form("distance_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        from_city = c1.text_input("出发城市*")
        to_city = c2.text_input("到达城市*")
        km = c3.number_input("公里数", min_value=0.0, value=0.0, step=1.0)
        if st.form_submit_button("新增 / 更新", type="primary"):
            if not from_city.strip() or not to_city.strip():
                st.error("出发城市、到达城市必填。")
            else:
                with db_session() as db:
                    a = from_city.strip()
                    b = to_city.strip()
                    rec = db.query(CityDistance).filter(CityDistance.from_city == a, CityDistance.to_city == b).first()
                    if rec:
                        rec.km = float(km)
                    else:
                        db.add(CityDistance(from_city=a, to_city=b, km=float(km)))
                    if not safe_commit(db, context=f"城市距离新增/更新：{a}->{b}"):
                        st.stop()
                clear_runtime_caches_after_data_change()
                st.success("已保存")
                st.rerun()

    with db_session() as db:
        dists = db.query(CityDistance).order_by(CityDistance.id.desc()).limit(300).all()
    rows = [{"ID": d.id, "from": d.from_city, "to": d.to_city, "km": round(float(d.km or 0), 1)} for d in dists]
    show_table(rows)
    if dists:
        delete_id = st.selectbox("删除距离记录（按ID）", [d.id for d in dists], key="delete_dist_select")
        if st.button("删除所选距离记录", key="delete_dist_btn"):
            with db_session() as db:
                obj = db.query(CityDistance).filter(CityDistance.id == delete_id).first()
                if obj:
                    db.delete(obj)
                    if not safe_commit(db, context=f"删除城市距离：dist#{delete_id}"):
                        st.stop()
            clear_runtime_caches_after_data_change()
            st.success("已删除")
            st.rerun()

# -------------------- 城市坐标 --------------------
elif page == "城市坐标":
    st.subheader("城市坐标")
    st.caption("用于自动计算全国城市直线距离；CSV 格式：name,lat,lon。")
    with st.form("city_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("城市名*")
        lat = c2.number_input("纬度 lat", value=0.0, step=0.000001, format="%.6f")
        lon = c3.number_input("经度 lon", value=0.0, step=0.000001, format="%.6f")
        if st.form_submit_button("新增 / 更新", type="primary"):
            if not name.strip():
                st.error("城市名必填。")
            else:
                with db_session() as db:
                    nm = name.strip()
                    rec = db.query(City).filter(City.name == nm).first()
                    if rec:
                        rec.lat = float(lat)
                        rec.lon = float(lon)
                    else:
                        db.add(City(name=nm, lat=float(lat), lon=float(lon)))
                    if not safe_commit(db, context=f"城市坐标新增/更新：{nm}"):
                        st.stop()
                clear_runtime_caches_after_data_change()
                st.success("已保存")
                st.rerun()

    csv_file = st.file_uploader("批量导入城市坐标 CSV", type=["csv"], key="city_csv")
    if st.button("执行 CSV 导入", key="city_csv_import_btn"):
        if not csv_file:
            st.warning("请先上传 CSV 文件。")
        else:
            text_ = csv_file.getvalue().decode("utf-8-sig", errors="ignore")
            reader = csv.reader(io.StringIO(text_))
            imported = 0
            with db_session() as db:
                for r in reader:
                    if not r or len(r) < 3:
                        continue
                    if str(r[0]).strip() in ("name", "城市", "city"):
                        continue
                    nm = str(r[0]).strip()
                    if not nm:
                        continue
                    try:
                        lat_v = float(r[1])
                        lon_v = float(r[2])
                    except Exception:
                        continue
                    rec = db.query(City).filter(City.name == nm).first()
                    if rec:
                        rec.lat = lat_v
                        rec.lon = lon_v
                    else:
                        db.add(City(name=nm, lat=lat_v, lon=lon_v))
                    imported += 1
                if not safe_commit(db, context="城市坐标 CSV 导入"):
                    st.stop()
            clear_runtime_caches_after_data_change()
            st.success(f"已导入 / 更新 {imported} 条城市坐标。")
            st.rerun()

    with db_session() as db:
        cities = db.query(City).order_by(City.id.desc()).limit(300).all()
    rows = [{"ID": c.id, "城市": c.name, "lat": round(float(c.lat), 6), "lon": round(float(c.lon), 6)} for c in cities]
    show_table(rows)
    if cities:
        delete_id = st.selectbox("删除城市（按ID）", [c.id for c in cities], key="delete_city_select")
        if st.button("删除所选城市", key="delete_city_btn"):
            with db_session() as db:
                obj = db.query(City).filter(City.id == delete_id).first()
                if obj:
                    db.delete(obj)
                    if not safe_commit(db, context=f"删除城市：city#{delete_id}"):
                        st.stop()
            clear_runtime_caches_after_data_change()
            st.success("已删除")
            st.rerun()

# -------------------- 模板导入 --------------------
elif page == "模板导入":
    st.subheader("模板导入")
    st.caption("下载模板 → 填写 → 上传导入，支持新增/更新。")

    def make_xlsx_template(headers, example_rows, sheet_name="template"):
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name
        ws.append(headers)
        for row in example_rows:
            ws.append(row)
        for i, h in enumerate(headers, start=1):
            col_letter = chr(64 + i) if i <= 26 else None
            if col_letter:
                ws.column_dimensions[col_letter].width = max(12, min(36, len(str(h)) * 2))
        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)
        return bio

    def read_xlsx_rows(uploaded_file):
        from openpyxl import load_workbook

        data = uploaded_file.getvalue()
        wb = load_workbook(io.BytesIO(data))
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return [], []
        headers = [str(x).strip() if x is not None else "" for x in rows[0]]
        out = []
        for r in rows[1:]:
            if not r or all(x is None or str(x).strip() == "" for x in r):
                continue
            first = str(r[0]).strip() if r[0] is not None else ""
            if first in ("必填", "说明", "字段说明"):
                continue
            out.append(list(r))
        return headers, out

    def find_idx(headers, aliases: list[str]) -> Optional[int]:
        for cand in aliases:
            for i, h in enumerate(headers):
                if str(h).strip() == str(cand).strip():
                    return i
        for cand in aliases:
            for i, h in enumerate(headers):
                if str(h).strip().startswith(str(cand).strip()):
                    return i
        return None

    headers_a = [
        "姓名", "性别(男/女)", "等级(A/B/C)", "可带队(是/否)", "常驻城市", "每周上限(院次)",
        "状态(在岗/请假/冻结)", "本月已排院次", "本月差旅天数", "连续工作天数",
        "上次结束城市(可空)", "上次结束日期(YYYY-MM-DD)(必填)",
    ]
    explain_a = ["必填", "默认女", "默认B", "默认是", "必填", "默认1", "默认在岗", "默认0", "默认0", "默认0", "可空", "必填"]
    example_a = [
        ["张三", "女", "A", "是", "北京", 1, "在岗", 0, 0, 0, "苏州", "2026-01-20"],
        ["李四", "女", "B", "是", "上海", 2, "在岗", 0, 0, 0, "", "2026-02-01"],
    ]
    bio_a = make_xlsx_template(headers_a, [explain_a] + example_a, sheet_name="稽查员")
    st.download_button("下载稽查员模板（XLSX）", bio_a.getvalue(), file_name="稽查员模板.xlsx", key="dl_aud_tpl")

    headers_t = [
        "项目名称", "客户/申办方", "需要A带队(是/否)", "所需人数", "任务天数", "性别要求(男/女/不限)",
        "硬指定人员(可空)", "软指定专家/老师(可空)", "中心城市", "开始日期(YYYY-MM-DD)", "结束日期(YYYY-MM-DD)(必填)",
    ]
    explain_t = ["必填", "可空", "默认否", "默认1", "默认1", "默认不限", "可空", "可空", "必填", "必填", "必填"]
    example_t = [
        ["项目A", "申办方X", "否", 2, 2, "不限", "", "张三", "苏州", "2026-02-01", "2026-02-02"],
        ["项目B", "申办方Y", "是", 1, 3, "女", "", "", "北京", "2026-02-03", "2026-02-05"],
    ]
    bio_t = make_xlsx_template(headers_t, [explain_t] + example_t, sheet_name="任务")
    st.download_button("下载任务模板（XLSX）", bio_t.getvalue(), file_name="任务模板.xlsx", key="dl_task_tpl")

    st.divider()
    auditor_xlsx = st.file_uploader("上传稽查员模板", type=["xlsx"], key="auditor_xlsx")
    if st.button("导入稽查员模板", key="import_aud_btn"):
        if not auditor_xlsx:
            st.warning("请先上传稽查员模板。")
        else:
            headers, rows = read_xlsx_rows(auditor_xlsx)
            imported = 0
            with db_session() as db:
                for r in rows:
                    name_i = find_idx(headers, ["姓名"])
                    base_i = find_idx(headers, ["常驻城市"])
                    if name_i is None or base_i is None:
                        continue
                    name = str(r[name_i] or "").strip()
                    base_city = str(r[base_i] or "").strip()
                    if not name or not base_city:
                        continue

                    gender_idx = find_idx(headers, ["性别(男/女)"])
                    group_idx = find_idx(headers, ["等级(A/B/C)"])
                    lead_idx = find_idx(headers, ["可带队(是/否)"])
                    week_idx = find_idx(headers, ["每周上限(院次)"])
                    status_idx = find_idx(headers, ["状态(在岗/请假/冻结)"])
                    month_idx = find_idx(headers, ["本月已排院次"])
                    travel_idx = find_idx(headers, ["本月差旅天数"])
                    cont_idx = find_idx(headers, ["连续工作天数"])
                    last_city_idx = find_idx(headers, ["上次结束城市(可空)"])
                    last_date_idx = find_idx(headers, ["上次结束日期(YYYY-MM-DD)(必填)"])

                    gender = str(r[gender_idx] if gender_idx is not None else "女")
                    group_level = str(r[group_idx] if group_idx is not None else "B")
                    can_lead_raw = str(r[lead_idx] if lead_idx is not None else "是")
                    last_date_raw = r[last_date_idx] if last_date_idx is not None else None
                    last_date = safe_parse_date(last_date_raw) or date.today()

                    rec = db.query(Auditor).filter(Auditor.name == name).first()
                    if rec:
                        rec.gender = gender or "女"
                        rec.group_level = group_level or "B"
                        rec.can_lead_team = can_lead_raw in BOOL_TRUE
                        rec.base_city = base_city
                        rec.max_weekly_tasks = _safe_int(r[week_idx], 1) if week_idx is not None else 1
                        rec.status = STATUS_MAP.get(str(r[status_idx]).strip(), "active") if status_idx is not None else "active"
                        rec.monthly_cases = _safe_int(r[month_idx], 0) if month_idx is not None else 0
                        rec.travel_days = _safe_int(r[travel_idx], 0) if travel_idx is not None else 0
                        rec.continuous_days = _safe_int(r[cont_idx], 0) if cont_idx is not None else 0
                        rec.last_task_end_city = str(r[last_city_idx]).strip() if last_city_idx is not None and r[last_city_idx] is not None else None
                        rec.last_task_end_date = last_date
                    else:
                        db.add(
                            Auditor(
                                name=name,
                                gender=gender or "女",
                                group_level=group_level or "B",
                                can_lead_team=can_lead_raw in BOOL_TRUE,
                                base_city=base_city,
                                max_weekly_tasks=_safe_int(r[week_idx], 1) if week_idx is not None else 1,
                                status=STATUS_MAP.get(str(r[status_idx]).strip(), "active") if status_idx is not None else "active",
                                monthly_cases=_safe_int(r[month_idx], 0) if month_idx is not None else 0,
                                travel_days=_safe_int(r[travel_idx], 0) if travel_idx is not None else 0,
                                continuous_days=_safe_int(r[cont_idx], 0) if cont_idx is not None else 0,
                                last_task_end_city=str(r[last_city_idx]).strip() if last_city_idx is not None and r[last_city_idx] is not None else None,
                                last_task_end_date=last_date,
                            )
                        )
                    imported += 1
                if not safe_commit(db, "导入稽查员模板"):
                    st.stop()
            clear_runtime_caches_after_data_change()
            st.success(f"已导入 / 更新 {imported} 条稽查员记录。")
            st.rerun()

    st.divider()
    task_xlsx = st.file_uploader("上传任务模板", type=["xlsx"], key="task_xlsx")
    if st.button("导入任务模板", key="import_task_btn"):
        if not task_xlsx:
            st.warning("请先上传任务模板。")
        else:
            headers, rows = read_xlsx_rows(task_xlsx)
            imported = 0
            with db_session() as db:
                for r in rows:
                    proj_i = find_idx(headers, ["项目名称"])
                    city_i = find_idx(headers, ["中心城市"])
                    sd_i = find_idx(headers, ["开始日期(YYYY-MM-DD)"])
                    ed_i = find_idx(headers, ["结束日期(YYYY-MM-DD)(必填)"])
                    if None in (proj_i, city_i, sd_i, ed_i):
                        continue

                    project_name = str(r[proj_i] or "").strip()
                    site_city = str(r[city_i] or "").strip()
                    start_d = safe_parse_date(r[sd_i])
                    end_d = safe_parse_date(r[ed_i])

                    if not project_name or not site_city or not start_d or not end_d:
                        continue
                    if end_d < start_d:
                        continue

                    customer_i = find_idx(headers, ["客户/申办方"])
                    need_i = find_idx(headers, ["需要A带队(是/否)"])
                    head_i = find_idx(headers, ["所需人数"])
                    days_i = find_idx(headers, ["任务天数"])
                    gender_i = find_idx(headers, ["性别要求(男/女/不限)"])
                    hard_i = find_idx(headers, ["硬指定人员(可空)"])
                    soft_i = find_idx(headers, ["软指定专家/老师(可空)"])

                    rec = db.query(Task).filter(Task.project_name == project_name, Task.start_date == start_d, Task.site_city == site_city).first()
                    if rec:
                        rec.customer_name = str(r[customer_i]).strip() if customer_i is not None and r[customer_i] is not None else None
                        rec.need_expert = str(r[need_i]).strip() in BOOL_TRUE if need_i is not None else False
                        rec.required_headcount = _safe_int(r[head_i], 1) if head_i is not None else 1
                        rec.required_days = _safe_int(r[days_i], max(1, (end_d - start_d).days + 1)) if days_i is not None else max(1, (end_d - start_d).days + 1)
                        rec.required_gender = str(r[gender_i]).strip() if gender_i is not None and r[gender_i] is not None else "不限"
                        rec.specified_auditors = str(r[hard_i]).strip() if hard_i is not None and r[hard_i] is not None else None
                        rec.preferred_experts = str(r[soft_i]).strip() if soft_i is not None and r[soft_i] is not None else None
                        rec.end_date = end_d
                    else:
                        db.add(
                            Task(
                                project_name=project_name,
                                customer_name=str(r[customer_i]).strip() if customer_i is not None and r[customer_i] is not None else None,
                                need_expert=str(r[need_i]).strip() in BOOL_TRUE if need_i is not None else False,
                                required_headcount=_safe_int(r[head_i], 1) if head_i is not None else 1,
                                required_days=_safe_int(r[days_i], max(1, (end_d - start_d).days + 1)) if days_i is not None else max(1, (end_d - start_d).days + 1),
                                required_gender=str(r[gender_i]).strip() if gender_i is not None and r[gender_i] is not None else "不限",
                                specified_auditors=str(r[hard_i]).strip() if hard_i is not None and r[hard_i] is not None else None,
                                preferred_experts=str(r[soft_i]).strip() if soft_i is not None and r[soft_i] is not None else None,
                                site_city=site_city,
                                start_date=start_d,
                                end_date=end_d,
                            )
                        )
                    imported += 1
                if not safe_commit(db, "导入任务模板"):
                    st.stop()
            clear_runtime_caches_after_data_change()
            st.success(f"已导入 / 更新 {imported} 条任务记录。")
            st.rerun()

# -------------------- 日历视图 --------------------
elif page == "日历视图":
    st.subheader("日历视图")
    st.caption("按月查看排班，支持折叠展示、月历图片下载与排班明细维护。")

    data_version = int(st.session_state.get("data_version", 0))
    c1, c2, c3 = st.columns(3)
    auditor_rows = get_auditors_for_ui(data_version)
    auditor_options = {"全部稽查员": None}
    for a in auditor_rows:
        auditor_options[f"#{a['id']} {a['name']}"] = a["id"]

    auditor_label = c1.selectbox("筛选稽查员", list(auditor_options.keys()), key="cal_auditor_filter")
    year = c2.selectbox("年份", list(range(date.today().year - 2, date.today().year + 3)), index=2, key="cal_year")
    month = c3.selectbox("月份", list(range(1, 13)), index=date.today().month - 1, key="cal_month")
    auditor_id = auditor_options[auditor_label]

    payload = get_calendar_payload(data_version, int(year), int(month), auditor_id)
    month_start = payload["month_start"]
    merged_rows = payload["merged_rows"]
    day_marks = payload["day_marks"]
    events_by_day = payload["events_by_day"]

    cal_png = build_calendar_png_bytes(int(year), int(month), events_by_day, day_marks)
    st.download_button(
        "下载当月日历图片PNG",
        data=cal_png,
        file_name=f"排班日历_{year}_{month:02d}.png",
        mime="image/png",
    )

    st.subheader(f"{year}年{month}月 日历总览")
    weeks = []
    first_cell = month_start - timedelta(days=month_start.weekday())
    current = first_cell
    for _ in range(6):
        row = []
        for _ in range(7):
            row.append(current)
            current += timedelta(days=1)
        weeks.append(row)

    headers = st.columns(7)
    for idx, h in enumerate(["周一", "周二", "周三", "周四", "周五", "周六", "周日"]):
        headers[idx].markdown(f"**{h}**")

    for week in weeks:
        cols = st.columns(7)
        for idx, day in enumerate(week):
            marks = []
            mk = day_marks.get(day.isoformat())
            if mk:
                marks.append(mk.get("label") or mk.get("type") or "标记")
            day_events = events_by_day.get(day.isoformat(), [])
            show_events = []
            for obj in day_events[:2]:
                txt = f"(定){obj['project']}｜{'、'.join(obj['persons'])}"
                if len(txt) > 26:
                    txt = txt[:26] + "…"
                show_events.append(txt)
            color = "#ffffff"
            if day.month != month:
                color = "#f7f7f7"
            elif show_events:
                color = "#eef6ff"
            cols[idx].markdown(
                f"<div style='border:1px solid #ddd;border-radius:10px;padding:8px;min-height:116px;background:{color};overflow:hidden;'>"
                f"<div style='font-weight:700;margin-bottom:2px'>{day.day}</div>"
                + (f"<div style='color:#16a34a;font-size:12px;margin-bottom:2px'>{' / '.join(marks)}</div>" if marks else "")
                + ("" if not show_events else "".join([f"<div style='font-size:12px;line-height:1.35;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{e}</div>" for e in show_events]))
                + (f"<div style='font-size:12px;color:#666;margin-top:4px'>还有 {len(day_events)-2} 项…</div>" if len(day_events) > 2 else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("本月排班明细")
    rows = []
    for s in merged_rows:
        rows.append(
            {
                "来源": "已定直录" if s.get("source") == "direct" else "标准排班",
                "记录ID": s.get("id"),
                "任务ID": s.get("task_id"),
                "项目": s.get("project_name") or "",
                "城市": s.get("site_city") or "",
                "角色": "组长" if s.get("role") == "leader" else "成员",
                "人员": s.get("auditor_name") or "",
                "开始日期": d2s(s.get("start_date")),
                "结束日期": d2s(s.get("end_date")),
            }
        )
    show_paginated_table(rows, "calendar_month_rows", 320, default_page_size=50)

    st.divider()
    st.subheader("修改已定项目人员明细")
    direct_task_options = {}
    with db_session() as db:
        all_tasks = db.query(Task).order_by(Task.id.desc()).all()
    for t in all_tasks:
        if get_direct_assignments(int(t.id)):
            direct_task_options[f"#{t.id} {t.project_name}｜{t.site_city}｜{d2s(t.start_date)}"] = t.id

    if not direct_task_options:
        st.info("当前没有已定项目人员记录。")
    else:
        direct_task_label = st.selectbox("选择任务", list(direct_task_options.keys()), key="edit_direct_task_select")
        direct_task_id = int(direct_task_options[direct_task_label])
        direct_assignments = get_direct_assignments(direct_task_id)
        edited = st.data_editor(
            pd.DataFrame([
                {
                    "项目名称": r.get("project_name", "") or "",
                    "类型": "兼职" if bool(r.get("is_part_time")) else "内部稽查员",
                    "人员姓名": r.get("person_name", "") or "",
                    "角色": "组长" if str(r.get("role", "")) == "leader" else "成员",
                    "开始日期": str(r.get("start_date", "")),
                    "结束日期": str(r.get("end_date", "")),
                    "备注": r.get("notes", "") or "",
                }
                for r in direct_assignments
            ]),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="direct_assignment_editor",
        )
        if st.button("保存已定项目人员修改", type="primary", key="save_direct_assignments_btn"):
            ok, msg = save_direct_assignments_from_df(direct_task_id, edited)
            if ok:
                clear_runtime_caches_after_data_change()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
