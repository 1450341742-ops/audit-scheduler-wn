import csv
import io
import json
import os
import re
import hashlib
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.db import Base, SessionLocal, engine, ensure_schema, IS_SQLITE, IS_SQLITE
from app.models import Auditor, Task, Schedule, CityDistance, City
from app.scheduler import (
    build_candidates,
    propose_team,
    compute_from_city,
    get_distance_km,
    team_objective,
)
from app.seed_distances import SEED_CITY_DISTANCES, CITY_COORDS

APP_NAME = "万宁睿和稽查排班"
st.set_page_config(page_title=APP_NAME, layout="wide")

# -------------------- 上传控件中文化 --------------------
st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzoneInstructions"]{
        display:none !important;
    }

    [data-testid="stFileUploaderDropzone"]{
        position: relative !important;
    }
    [data-testid="stFileUploaderDropzone"]::before{
        content:"将文件拖拽到此处，或点击右侧“浏览文件”上传（支持 .xlsx/.csv，单个文件 ≤200MB）";
        display:block;
        color:#333;
        padding:10px 6px 8px 6px;
        line-height:1.5;
        font-size:14px;
        white-space:normal;
    }

    [data-testid="stFileUploaderDropzone"] input[type="file"]{
        width: 100% !important;
    }

    [data-testid="stFileUploaderDropzone"] input[type="file"]::file-selector-button{
        min-width: 132px !important;
        height: 42px !important;
        padding: 0 18px !important;
        border-radius: 10px !important;
        color: transparent !important;
        -webkit-text-fill-color: transparent !important;
        text-shadow: none !important;
    }

    [data-testid="stFileUploaderDropzone"]::after{
        content:"浏览文件";
        position:absolute;
        right: 14px;
        top: 50%;
        transform: translateY(-50%);
        min-width: 132px;
        height: 42px;
        padding: 0 18px;
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.15);
        background: rgba(255,255,255,0.96);
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:14px;
        font-weight:600;
        color:#111;
        white-space:nowrap;
        pointer-events:none;
        z-index: 9999;
        box-sizing: border-box;
    }

    @media (max-width: 520px){
        [data-testid="stFileUploaderDropzone"] input[type="file"]::file-selector-button{
            min-width:124px !important;
            height:40px !important;
            padding:0 14px !important;
        }
        [data-testid="stFileUploaderDropzone"]::after{
            min-width:124px;
            height:40px;
            padding:0 14px;
            right: 12px;
            font-size:14px;
        }
        [data-testid="stFileUploaderDropzone"]::before{
            font-size:13px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    ensure_extra_tables
except NameError:
    def ensure_extra_tables():
        try:
            ensure_support_tables()
        except Exception:
            return None


# -------------------- 初始化 --------------------


@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def safe_parse_date(value) -> Optional[date]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    try:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().date()
    except Exception:
        pass

    try:
        if isinstance(value, (int, float)) and not (isinstance(value, float) and pd.isna(value)):
            base = datetime(1899, 12, 30)
            return (base + timedelta(days=float(value))).date()

        s_num = str(value).strip()
        if re.fullmatch(r"\d+(\.\d+)?", s_num):
            base = datetime(1899, 12, 30)
            return (base + timedelta(days=float(s_num))).date()
    except Exception:
        pass

    s = str(value).strip()
    if not s:
        return None

    if " " in s:
        s = s.split(" ")[0].strip()
    s = s.replace("/", "-").replace(".", "-")

    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y%m%d"):
        try:
            d = datetime.strptime(s, fmt).date()
            if fmt == "%Y-%m":
                return d.replace(day=1)
            return d
        except Exception:
            pass

    return None


def d2s(v: Optional[date]) -> str:
    return v.strftime("%Y-%m-%d") if v else ""


def date_ranges_overlap(a_start, a_end, b_start, b_end) -> bool:
    if a_start is None or b_start is None:
        return False
    a_end = a_end or a_start
    b_end = b_end or b_start
    return not (a_end < b_start or b_end < a_start)



def show_table(rows: list[dict], height: int = 380):
    if not rows:
        st.info("暂无数据")
        return
    st.dataframe(rows, use_container_width=True, height=height)


def safe_commit(db: Session, context: str = "") -> bool:
    try:
        db.commit()
        return True
    except IntegrityError as e:
        db.rollback()
        st.error(f"数据库写入失败：{context}。常见原因：重复数据 / 唯一约束冲突。")
        st.exception(e)
        return False
    except Exception as e:
        db.rollback()
        st.error(f"数据库写入失败：{context}")
        st.exception(e)
        return False


def clear_runtime_caches_after_data_change():
    st.session_state["_data_version"] = int(st.session_state.get("_data_version", 0) or 0) + 1
    try:
        st.cache_data.clear()
    except Exception:
        pass
    for k in [
        "recommend_result",
        "batch_report",
        "smart_task_select",
        "member_ids_text",
        "edit_auditor_select",
        "edit_task_select",
    ]:
        if k in st.session_state:
            st.session_state.pop(k, None)


def get_data_version() -> int:
    return int(st.session_state.get("_data_version", 0) or 0)


def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, float) and pd.isna(x):
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def normalize_text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


# -------------------- 扩展功能支持表 / 缺失函数修复 --------------------
def ensure_support_tables():
    with engine.begin() as conn:
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS progress_targets (
                id INTEGER PRIMARY KEY,
                period_type TEXT NOT NULL,
                year INTEGER NOT NULL,
                period_value INTEGER NOT NULL DEFAULT 0,
                target_projects INTEGER NOT NULL DEFAULT 0,
                target_staffing INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT
            )
        '''))
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS part_time_staff (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                base_city TEXT,
                note TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT
            )
        '''))
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS direct_assignments (
                id INTEGER PRIMARY KEY,
                task_id INTEGER NOT NULL,
                auditor_id INTEGER,
                person_name TEXT NOT NULL,
                is_part_time INTEGER NOT NULL DEFAULT 0,
                role TEXT NOT NULL DEFAULT 'member',
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                notes TEXT,
                created_at TEXT
            )
        '''))
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS auditor_monthly_targets (
                auditor_id INTEGER PRIMARY KEY,
                monthly_target INTEGER NOT NULL DEFAULT 4,
                updated_at TEXT
            )
        '''))
        try:
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_progress_targets_period ON progress_targets(period_type, year, period_value)"))
        except Exception:
            pass


def parse_name_list(raw) -> list[str]:
    s = normalize_text(raw)
    if not s:
        return []
    for sep in ["，", "、", ";", "；", "/", "|", "\n"]:
        s = s.replace(sep, ",")
    out = []
    seen = set()
    for x in s.split(","):
        x = x.strip()
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def get_part_time_staff_rows(active_only: bool = False) -> list[dict]:
    ensure_support_tables()
    with engine.begin() as conn:
        sql = "SELECT id, name, base_city, note, is_active, created_at FROM part_time_staff"
        if active_only:
            sql += " WHERE is_active=1"
        sql += " ORDER BY id DESC"
        rows = conn.execute(text(sql)).mappings().all()
    return [dict(r) for r in rows]


def save_part_time_staff(name: str, base_city: str = '', note: str = '', is_active: bool = True):
    ensure_support_tables()
    clean_name = normalize_text(name)
    if not clean_name:
        return False, '兼职姓名不能为空'
    params = {
        'name': clean_name,
        'base_city': normalize_text(base_city),
        'note': normalize_text(note),
        'is_active': 1 if is_active else 0,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with engine.begin() as conn:
        exists = conn.execute(text('SELECT id FROM part_time_staff WHERE name=:name'), {'name': clean_name}).mappings().first()
        if exists:
            conn.execute(text('UPDATE part_time_staff SET base_city=:base_city, note=:note, is_active=:is_active WHERE id=:id'), {**params, 'id': int(exists['id'])})
            return True, '兼职人员已更新'
        next_id = conn.execute(text('SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM part_time_staff')).mappings().first()['next_id']
        conn.execute(text('INSERT INTO part_time_staff (id, name, base_city, note, is_active, created_at) VALUES (:id, :name, :base_city, :note, :is_active, :created_at)'), {**params, 'id': int(next_id)})
    return True, '兼职人员已保存'


def delete_part_time_staff(row_id: int):
    ensure_support_tables()
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM part_time_staff WHERE id=:id'), {'id': int(row_id)})
    return True, '兼职人员已删除'


def get_direct_assignments(task_id: int) -> list[dict]:
    ensure_support_tables()
    with engine.begin() as conn:
        rows = conn.execute(text('SELECT id, task_id, auditor_id, person_name, is_part_time, role, start_date, end_date, notes, created_at FROM direct_assignments WHERE task_id=:task_id ORDER BY id ASC'), {'task_id': int(task_id)}).mappings().all()
    out = []
    for r in rows:
        item = dict(r)
        item['is_part_time'] = bool(int(item.get('is_part_time') or 0))
        item['start_date'] = item.get('start_date') or ''
        item['end_date'] = item.get('end_date') or ''
        out.append(item)
    return out


def replace_direct_assignments(task_id: int, rows_to_save: list[dict]):
    ensure_support_tables()
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM direct_assignments WHERE task_id=:task_id'), {'task_id': int(task_id)})
        next_id_row = conn.execute(text('SELECT COALESCE(MAX(id), 0) AS max_id FROM direct_assignments')).mappings().first()
        next_id = int(next_id_row['max_id'] or 0) + 1
        for r in rows_to_save or []:
            sd = safe_parse_date(r.get('start_date'))
            ed = safe_parse_date(r.get('end_date'))
            if not sd or not ed:
                continue
            conn.execute(text('''
                INSERT INTO direct_assignments (id, task_id, auditor_id, person_name, is_part_time, role, start_date, end_date, notes, created_at)
                VALUES (:id, :task_id, :auditor_id, :person_name, :is_part_time, :role, :start_date, :end_date, :notes, :created_at)
            '''), {
                'id': next_id,
                'task_id': int(task_id),
                'auditor_id': r.get('auditor_id'),
                'person_name': normalize_text(r.get('person_name')),
                'is_part_time': 1 if bool(r.get('is_part_time')) else 0,
                'role': 'leader' if str(r.get('role')) == 'leader' else 'member',
                'start_date': d2s(sd),
                'end_date': d2s(ed),
                'notes': normalize_text(r.get('notes')),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
            next_id += 1
    return True


def sync_task_schedules_from_direct_assignments(task: Task):
    ensure_support_tables()
    direct_rows = get_direct_assignments(int(task.id))
    if not direct_rows:
        return False, '未找到已定项目人员'

    with db_session() as db:
        db.query(Schedule).filter(Schedule.task_id == int(task.id)).delete()
        for r in direct_rows:
            if bool(r.get('is_part_time')):
                continue
            auditor_id = r.get('auditor_id')
            if not auditor_id:
                auditor = db.query(Auditor).filter(Auditor.name == normalize_text(r.get('person_name'))).first()
                auditor_id = int(auditor.id) if auditor else None
            else:
                auditor = db.query(Auditor).filter(Auditor.id == int(auditor_id)).first()
            if not auditor_id or not auditor:
                continue
            sd = safe_parse_date(r.get('start_date')) or task.start_date
            ed = safe_parse_date(r.get('end_date')) or task.end_date or task.start_date
            from_city = compute_from_city(auditor, task)
            km = get_distance_km(db, from_city, task.site_city)
            db.add(Schedule(
                task_id=int(task.id),
                auditor_id=int(auditor_id),
                role='leader' if str(r.get('role')) == 'leader' else 'member',
                start_date=sd,
                end_date=ed,
                travel_from_city=from_city,
                travel_to_city=task.site_city,
                distance_km=float(km),
                score=0.0,
                status='confirmed',
            ))
        ok = safe_commit(db, f'同步已定项目排班#{task.id}')
        if not ok:
            return False, '同步失败'
    return True, '已按已定项目人员直接录入排班'


def auto_fill_direct_assignments_from_specified(task_id: int, overwrite: bool = True):
    ensure_support_tables()
    with db_session() as db:
        task = db.query(Task).filter(Task.id == int(task_id)).first()
        if not task:
            return False, "任务不存在"
        specified_names = parse_name_list(getattr(task, "specified_auditors", None))
        if not specified_names:
            return False, "未填写硬指定人员"

        rows_to_save = []
        for idx, name in enumerate(specified_names):
            auditor = db.query(Auditor).filter(Auditor.name == normalize_text(name)).first()
            rows_to_save.append(
                {
                    "auditor_id": int(auditor.id) if auditor else None,
                    "person_name": normalize_text(name),
                    "is_part_time": False if auditor else True,
                    "role": "leader" if idx == 0 else "member",
                    "start_date": task.start_date,
                    "end_date": task.end_date or task.start_date,
                    "notes": "由硬指定人员自动生成",
                }
            )

    existing = get_direct_assignments(int(task_id))
    if existing and not overwrite:
        with db_session() as db:
            task = db.query(Task).filter(Task.id == int(task_id)).first()
            if task:
                sync_task_schedules_from_direct_assignments(task)
        return True, "已存在已定项目人员，并已同步到排班记录"

    replace_direct_assignments(int(task_id), rows_to_save)
    with db_session() as db:
        task = db.query(Task).filter(Task.id == int(task_id)).first()
        if task:
            sync_task_schedules_from_direct_assignments(task)
    return True, "已根据硬指定人员自动生成直录排班，并同步到排班记录"


def _get_period_range(period_type: str, year: int, period_value: int):
    if period_type == 'monthly':
        start_d = date(year, int(period_value), 1)
        if int(period_value) == 12:
            end_d = date(year, 12, 31)
        else:
            end_d = date(year, int(period_value) + 1, 1) - timedelta(days=1)
    elif period_type == 'quarterly':
        q = max(1, min(4, int(period_value or 1)))
        start_month = (q - 1) * 3 + 1
        start_d = date(year, start_month, 1)
        if q == 4:
            end_d = date(year, 12, 31)
        else:
            end_d = date(year, start_month + 3, 1) - timedelta(days=1)
    else:
        start_d = date(year, 1, 1)
        end_d = date(year, 12, 31)
    return start_d, end_d


def get_target_row(period_type: str, year: int, period_value: int) -> dict:
    ensure_support_tables()
    with engine.begin() as conn:
        row = conn.execute(text('SELECT period_type, year, period_value, target_projects, target_staffing, updated_at FROM progress_targets WHERE period_type=:period_type AND year=:year AND period_value=:period_value'), {'period_type': period_type, 'year': int(year), 'period_value': int(period_value or 0)}).mappings().first()
    if row:
        return dict(row)
    return {'period_type': period_type, 'year': int(year), 'period_value': int(period_value or 0), 'target_projects': 0, 'target_staffing': 0}


def save_target_row(period_type: str, year: int, period_value: int, target_projects: int, target_staffing: int):
    ensure_support_tables()
    params = {'period_type': period_type, 'year': int(year), 'period_value': int(period_value or 0), 'target_projects': int(target_projects or 0), 'target_staffing': int(target_staffing or 0), 'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    with engine.begin() as conn:
        row = conn.execute(text('SELECT id FROM progress_targets WHERE period_type=:period_type AND year=:year AND period_value=:period_value'), params).mappings().first()
        if row:
            conn.execute(text('UPDATE progress_targets SET target_projects=:target_projects, target_staffing=:target_staffing, updated_at=:updated_at WHERE id=:id'), {**params, 'id': int(row['id'])})
        else:
            next_id = conn.execute(text('SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM progress_targets')).mappings().first()['next_id']
            conn.execute(text('INSERT INTO progress_targets (id, period_type, year, period_value, target_projects, target_staffing, updated_at) VALUES (:id, :period_type, :year, :period_value, :target_projects, :target_staffing, :updated_at)'), {**params, 'id': int(next_id)})
    return True


def get_period_completed_counts(period_type: str, year: int, period_value: int):
    start_d, end_d = _get_period_range(period_type, int(year), int(period_value or 0))
    today = date.today()
    with db_session() as db:
        tasks = db.query(Task).filter(Task.start_date >= start_d, Task.start_date <= end_d).order_by(Task.start_date.asc(), Task.id.asc()).all()
    recorded_tasks = list(tasks)
    completed_tasks = [t for t in recorded_tasks if (t.end_date or t.start_date) < today]
    return start_d, end_d, recorded_tasks, completed_tasks


def get_monthly_target_value(year: int, month: int) -> int:
    row = get_target_row('monthly', int(year), int(month))
    return int(row.get('target_projects', 0) or 0)


def get_annual_monthly_progress(year: int) -> pd.DataFrame:
    rows = []
    for m in range(1, 13):
        start_d, end_d = _get_period_range('monthly', int(year), int(m))
        today = date.today()
        with db_session() as db:
            tasks = db.query(Task).filter(Task.start_date >= start_d, Task.start_date <= end_d).all()
        actual = sum(1 for t in tasks if (t.end_date or t.start_date) < today)
        plan = get_monthly_target_value(int(year), int(m))
        rows.append({'月份': f'{m}月', '实际完成': int(actual), '计划完成': int(plan)})
    return pd.DataFrame(rows)


def get_auditor_monthly_target_map() -> dict[int, int]:
    ensure_support_tables()
    with engine.begin() as conn:
        rows = conn.execute(text('SELECT auditor_id, monthly_target FROM auditor_monthly_targets')).mappings().all()
    return {int(r['auditor_id']): int(r.get('monthly_target') or 4) for r in rows}


def save_auditor_monthly_target(auditor_id: int, monthly_target: int):
    ensure_support_tables()
    params = {
        'auditor_id': int(auditor_id),
        'monthly_target': int(monthly_target or 0),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with engine.begin() as conn:
        row = conn.execute(text('SELECT auditor_id FROM auditor_monthly_targets WHERE auditor_id=:auditor_id'), params).mappings().first()
        if row:
            conn.execute(text('UPDATE auditor_monthly_targets SET monthly_target=:monthly_target, updated_at=:updated_at WHERE auditor_id=:auditor_id'), params)
        else:
            conn.execute(text('INSERT INTO auditor_monthly_targets (auditor_id, monthly_target, updated_at) VALUES (:auditor_id, :monthly_target, :updated_at)'), params)
    return True


def get_auditor_period_stats(period_type: str, year: int, period_value: int) -> pd.DataFrame:
    start_d, end_d = _get_period_range(period_type, int(year), int(period_value or 0))
    today = date.today()
    months_multiplier = {'monthly': 1, 'quarterly': 3, 'yearly': 12}.get(period_type, 1)
    target_map = get_auditor_monthly_target_map()
    with db_session() as db:
        auditors = db.query(Auditor).order_by(Auditor.id.asc()).all()
        schedules = db.query(Schedule).join(Task, Task.id == Schedule.task_id).filter(Task.start_date >= start_d, Task.start_date <= end_d).all()
        task_map = {int(t.id): t for t in db.query(Task).filter(Task.start_date >= start_d, Task.start_date <= end_d).all()}
    completed_pairs = set()
    for s in schedules:
        task = task_map.get(int(s.task_id))
        if not task:
            continue
        task_end = task.end_date or task.start_date
        if task_end < today:
            completed_pairs.add((int(s.auditor_id), int(s.task_id)))
    completed_by_auditor = {}
    for aid, tid in completed_pairs:
        completed_by_auditor[aid] = completed_by_auditor.get(aid, 0) + 1
    rows = []
    for a in auditors:
        monthly_target = int(target_map.get(int(a.id), 4) or 4)
        plan = monthly_target * months_multiplier
        actual = int(completed_by_auditor.get(int(a.id), 0))
        rows.append({
            '稽查员': a.name,
            '月度计划': monthly_target,
            '本周期计划': int(plan),
            '本周期实际完成': int(actual),
            '完成率': round(actual / plan * 100, 1) if plan else 0.0,
        })
    return pd.DataFrame(rows)


def get_progress_stats(period_type: str, year: int, period_value: int):
    start_d, end_d = get_period_date_range(period_type, year, period_value)
    today_d = date.today()

    target_row = get_target_row(period_type, year, period_value)
    target_projects = int(target_row.get("target_projects", 0) or 0)
    target_staffing = int(target_row.get("target_staffing", 0) or 0)

    with db_session() as db:
        tasks = db.query(Task).all()
        schedules = db.query(Schedule).all()

    tasks_in_period = []
    for t in tasks:
        t_start = getattr(t, "start_date", None)
        t_end = getattr(t, "end_date", None) or t_start
        if date_ranges_overlap(t_start, t_end, start_d, end_d):
            tasks_in_period.append(t)

    recorded_projects = len({int(t.id) for t in tasks_in_period})
    completed_projects = len({
        int(t.id) for t in tasks_in_period
        if (getattr(t, "end_date", None) or getattr(t, "start_date", None)) and (getattr(t, "end_date", None) or getattr(t, "start_date", None)) < today_d
    })

    staffing_actual = 0
    for s in schedules:
        s_start = getattr(s, "start_date", None)
        s_end = getattr(s, "end_date", None) or s_start
        if date_ranges_overlap(s_start, s_end, start_d, end_d):
            staffing_actual += 1

    project_completion_rate = round((completed_projects / target_projects * 100.0), 1) if target_projects else 0.0
    staffing_completion_rate = round((staffing_actual / target_staffing * 100.0), 1) if target_staffing else 0.0

    return {
        "start_date": start_d,
        "end_date": end_d,
        "target_projects": target_projects,
        "target_staffing": target_staffing,
        "recorded_projects": recorded_projects,
        "completed_projects": completed_projects,
        "staffing_actual": staffing_actual,
        "project_completion_rate": project_completion_rate,
        "staffing_completion_rate": staffing_completion_rate,
    }

# -------------------- 智能排班 --------------------
if page == "智能排班":
    st.subheader("智能排班")
    st.caption("先按硬约束筛选，再按距离优先 + 适度负荷均衡评分推荐。")

    payload = load_smart_page_payload(get_data_version())
    tasks = payload["tasks"]
    schedules_recent_rows = payload["schedules_recent_rows"]

    if not tasks:
        st.info("请先在【任务管理】中录入任务。")
    else:
        task_options = {
            f"#{t['id']} {t['project_name']}｜{t['site_city']}｜{t['start_date']}｜{t['required_days']}天｜{t['required_headcount']}人": t['id']
            for t in tasks
        }
        selected_label = st.selectbox("选择任务", list(task_options.keys()), key="smart_task_select")
        selected_task_id = task_options[selected_label]

        col_a, _ = st.columns([1, 3])
        if col_a.button("生成推荐", type="primary", key="gen_reco_btn"):
            with db_session() as db:
                task = db.query(Task).filter(Task.id == selected_task_id).first()
                auditors = db.query(Auditor).all()
                schedules_all = db.query(Schedule).all()
                candidates = build_candidates(db, task, auditors, schedules_all) if task else []
                team = propose_team(task, candidates) if task else None
                st.session_state["recommend_result"] = {
                    "task_id": selected_task_id,
                    "candidates": candidates[:25],
                    "team": team,
                    "error": None if team else "无可用团队方案",
                }
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
            team = rec.get("team")
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
                for i, c in enumerate(cands, start=1):
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
    show_table(schedules_recent_rows, 360)
    if schedules_recent_rows:
        delete_sid = st.selectbox("删除排班记录（按ID）", [r["ID"] for r in schedules_recent_rows], key="delete_schedule_select")
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

    show_table(rows, 320)

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
        default_end = start_date + timedelta(days=max(1, int(required_days)) - 1)
        end_date = st.date_input("结束日期*", value=default_end)

        if st.form_submit_button("新增任务", type="primary"):
            if not project_name.strip() or not site_city.strip():
                st.error("项目名称、中心城市必填。")
            elif end_date < start_date:
                st.error("结束日期不能早于开始日期。")
            else:
                with db_session() as db:
                    db.add(
                        Task(
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
                    )
                    if not safe_commit(db, context=f"新增任务：{project_name.strip()}"):
                        st.stop()
                    new_task_id = int(db.query(Task.id).order_by(Task.id.desc()).first()[0])
                if specified.strip():
                    auto_fill_direct_assignments_from_specified(new_task_id, overwrite=True)
                clear_runtime_caches_after_data_change()
                st.success("已新增")
                st.rerun()

    with db_session() as db:
        tasks = db.query(Task).order_by(Task.id.desc()).all()
        auditors = db.query(Auditor).order_by(Auditor.name.asc()).all()

    rows = []
    for t in tasks:
        rows.append(
            {
                "ID": t.id,
                "项目": t.project_name,
                "客户": t.customer_name or "",
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

    show_table(rows, 320)

    if tasks:
        task_options = {
            f"#{t.id} {t.project_name}｜{t.site_city}｜{d2s(t.start_date)}": t.id
            for t in tasks
        }
        selected_task_label = st.selectbox("选择要编辑的任务", list(task_options.keys()), key="edit_task_select")
        selected_task_id = task_options[selected_task_label]
        selected_task = next((t for t in tasks if t.id == selected_task_id), None)

        if selected_task:
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
                    direct_df = pd.DataFrame(columns=["类型", "人员姓名", "角色", "开始日期", "结束日期", "备注"])

            with st.form("direct_assign_form", clear_on_submit=False):
                edited_direct = st.data_editor(
                    direct_df,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="dynamic",
                    key=f"direct_assign_editor_{selected_task.id}",
                    column_config={
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
                            "notes": str(r.get("备注", "")).strip(),
                        }
                    )
                return rows_out

            if save_direct:
                rows_to_save = _normalize_direct_rows(edited_direct)
                replace_direct_assignments(int(selected_task.id), rows_to_save)
                ok, msg = sync_task_schedules_from_direct_assignments(selected_task)
                clear_runtime_caches_after_data_change()
                if ok:
                    st.success("已定项目人员已保存，并已同步到日历排班")
                else:
                    st.warning("已定项目人员已保存，但同步到日历排班失败：" + str(msg))
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
    st.caption("支持录入院次指标数量，并自动统计已录入项目、已完成项目、年度月度趋势，以及每位稽查员月/季/年的计划与实际完成情况。")

    c1, c2, c3 = st.columns(3)
    period_type = c1.selectbox("统计周期", ["monthly", "quarterly", "yearly"], format_func=lambda x: {"monthly":"月度","quarterly":"季度","yearly":"年度"}[x])
    year = c2.number_input("年份", min_value=2024, max_value=2035, value=date.today().year, step=1)
    if period_type == "monthly":
        period_value = c3.selectbox("月份", list(range(1, 13)), index=max(0, date.today().month - 1))
    elif period_type == "quarterly":
        period_value = c3.selectbox("季度", [1, 2, 3, 4], index=(date.today().month - 1)//3)
    else:
        period_value = 0
        c3.markdown("**全年**")

    target = get_target_row(period_type, int(year), int(period_value))
    with st.form("target_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        target_projects = c1.number_input("院次指标数量", min_value=0, value=int(target.get("target_projects", 0) or 0), step=1)
        target_staffing = c2.number_input("人员院次安排数量", min_value=0, value=int(target.get("target_staffing", 0) or 0), step=1)
        if st.form_submit_button("保存指标", type="primary"):
            save_target_row(period_type, int(year), int(period_value), int(target_projects), int(target_staffing))
            st.success("指标已保存")
            st.rerun()

    start_d, end_d, recorded_projects, completed_projects, actual_staffing, detail_rows = get_progress_stats(period_type, int(year), int(period_value))
    target = get_target_row(period_type, int(year), int(period_value))
    t_projects = int(target.get("target_projects", 0) or 0)
    t_staff = int(target.get("target_staffing", 0) or 0)
    complete_rate = round(completed_projects / t_projects * 100, 1) if t_projects else 0.0
    staffing_rate = round(actual_staffing / t_staff * 100, 1) if t_staff else 0.0

    st.write(f"统计区间：{d2s(start_d)} ~ {d2s(end_d)}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("院次目标数", t_projects)
    m2.metric("已录入院次数", recorded_projects)
    m3.metric("已完成院次数", completed_projects)
    m4.metric("院次完成率", f"{complete_rate}%")

    summary_df = pd.DataFrame([
        {"指标": "院次目标数", "目标": t_projects, "实际": completed_projects, "完成率": f"{complete_rate}%"},
        {"指标": "人员院次安排数", "目标": t_staff, "实际": actual_staffing, "完成率": f"{staffing_rate}%"},
        {"指标": "已录入院次数", "目标": recorded_projects, "实际": recorded_projects, "完成率": "100.0%"},
    ])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.bar_chart(summary_df.set_index("指标")[["目标", "实际"]])

    if period_type == "yearly":
        st.subheader("年度视图：每月计划完成 vs 实际完成")
        annual_df = get_annual_monthly_progress(int(year))
        st.dataframe(annual_df, use_container_width=True, hide_index=True)
        st.bar_chart(annual_df.set_index("月份")[["计划完成", "实际完成"]])
        st.line_chart(annual_df.set_index("月份")[["计划完成", "实际完成"]])

    st.subheader("稽查员计划与实际完成对比")
    with db_session() as db:
        auditors = db.query(Auditor).order_by(Auditor.id.asc()).all()
    target_map = get_auditor_monthly_target_map()
    if auditors:
        plan_rows = []
        for a in auditors:
            plan_rows.append({
                '稽查员ID': int(a.id),
                '稽查员': a.name,
                '每月计划院次数': int(target_map.get(int(a.id), 4) or 4),
            })
        edit_df = st.data_editor(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True, num_rows='fixed', key='auditor_plan_editor')
        if st.button('保存每人月度计划院次数'):
            for _, r in pd.DataFrame(edit_df).iterrows():
                save_auditor_monthly_target(int(r['稽查员ID']), int(r['每月计划院次数']))
            st.success('已保存每位稽查员月度计划院次数')
            st.rerun()

    auditor_stats_df = get_auditor_period_stats(period_type, int(year), int(period_value))
    if not auditor_stats_df.empty:
        st.dataframe(auditor_stats_df, use_container_width=True, hide_index=True)
        chart_df = auditor_stats_df.set_index('稽查员')[["本周期计划", "本周期实际完成"]]
        st.bar_chart(chart_df)
        st.line_chart(chart_df)
    else:
        st.info("暂无稽查员统计数据")

    st.subheader("项目完成进度及人员院次安排进度明细")
    if detail_rows:
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    else:
        st.info("该统计区间暂无项目数据")

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


def render_day_detail_panel(calendar_event_map: dict):
    st.markdown("### 当天安排详情")
    day_options = sorted(calendar_event_map.keys())
    if not day_options:
        st.info("本月暂无安排")
        return
    selected_day = st.selectbox(
        "点击查看某一天的详细安排",
        options=day_options,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        key="calendar_detail_day_select",
    )
    items = calendar_event_map.get(selected_day, [])
    if not items:
        st.info("当天暂无安排")
        return
    detail_rows = []
    for item in items:
        detail_rows.append({
            "日期": selected_day.strftime("%Y-%m-%d"),
            "项目": item.get("project_name", "") if isinstance(item, dict) else str(item),
            "城市": item.get("site_city", "") if isinstance(item, dict) else "",
            "人员": (item.get("auditor_name", "") or item.get("person_name", "")) if isinstance(item, dict) else "",
            "角色": item.get("role", "") if isinstance(item, dict) else "",
            "开始": (d2s(item.get("start_date")) if not isinstance(item.get("start_date"), str) else item.get("start_date")) if isinstance(item, dict) else "",
            "结束": (d2s(item.get("end_date")) if not isinstance(item.get("end_date"), str) else item.get("end_date")) if isinstance(item, dict) else "",
            "来源": item.get("source", "") if isinstance(item, dict) else "",
        })
    st.dataframe(detail_rows, use_container_width=True, height=min(420, 70 + len(detail_rows) * 35))


# -------------------- 日历视图 --------------------
if page == "日历视图":
    st.subheader("日历视图")
    st.caption("按月查看排班、节假日标识，并支持导出 ICS 日历。")

    c1, c2, c3 = st.columns(3)
    with db_session() as db:
        auditors = db.query(Auditor).order_by(Auditor.name.asc()).all()

    auditor_options = {"全部稽查员": None}
    for a in auditors:
        auditor_options[f"#{a.id} {a.name}"] = a.id

    auditor_label = c1.selectbox("筛选稽查员", list(auditor_options.keys()), key="cal_auditor_filter")
    year = c2.selectbox("年份", list(range(date.today().year - 2, date.today().year + 3)), index=2, key="cal_year")
    month = c3.selectbox("月份", list(range(1, 13)), index=date.today().month - 1, key="cal_month")
    auditor_id = auditor_options[auditor_label]

    cal_payload = load_calendar_page_payload(get_data_version(), int(year), int(month), auditor_id)
    month_start = cal_payload["month_start"]
    month_end = cal_payload["month_end"]
    direct_rows_raw = cal_payload["direct_rows_raw"]
    all_schedules_rows = cal_payload["all_schedules_rows"]

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
            key = (cur.isoformat(), int(s.get("task_id")))
            events_by_day.setdefault(key, {"project": s.get("project_name") or f"任务#{s.get('task_id')}", "task_id": s.get("task_id"), "persons": [], "city": s.get("site_city") or ""})
            nm = s.get("auditor_name") or ""
            if nm and nm not in events_by_day[key]["persons"]:
                events_by_day[key]["persons"].append(nm)
            cur += timedelta(days=1)

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

            evs = []
            for (day_iso, task_id), obj in events_by_day.items():
                if day_iso != day.isoformat():
                    continue
                evs.append(f"#{task_id} {obj['project']}｜{'、'.join(obj['persons'])}")

            color = "#ffffff"
            if day.month != month:
                color = "#f7f7f7"
            elif evs:
                color = "#eef6ff"

            cols[idx].markdown(
                f"<div style='border:1px solid #ddd;border-radius:8px;padding:8px;min-height:120px;background:{color};'>"
                f"<div style='font-weight:600'>{day.day}</div>"
                + (f"<div style='color:#d97706;font-size:12px'>{' / '.join(marks)}</div>" if marks else "")
                + ("" if not evs else "".join([f"<div style='font-size:12px;margin-top:4px'>{e}</div>" for e in evs[:3]]))
                + (f"<div style='font-size:12px;color:#666'>还有 {len(evs)-3} 项</div>" if len(evs) > 3 else "")
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
    show_table(rows, 320)

    st.divider()
    st.subheader("修改已定项目人员明细")
    direct_task_options = {}
    with db_session() as db:
        all_tasks = db.query(Task).order_by(Task.id.desc()).all()
    for t in all_tasks:
        if get_direct_assignments(int(t.id)):
            direct_task_options[f"#{t.id} {t.project_name}｜{t.site_city}｜{d2s(t.start_date)}"] = int(t.id)

    if direct_task_options:
        dtask_label = st.selectbox("选择已定项目任务", list(direct_task_options.keys()), key="calendar_direct_task_select")
        dtask_id = direct_task_options[dtask_label]
        dtask = next((t for t in all_tasks if int(t.id) == int(dtask_id)), None)
        existing = get_direct_assignments(int(dtask_id))
        df = pd.DataFrame([
            {
                "类型": "兼职" if bool(r.get("is_part_time")) else "内部稽查员",
                "人员姓名": r.get("person_name", ""),
                "角色": "组长" if str(r.get("role")) == "leader" else "成员",
                "开始日期": str(r.get("start_date")),
                "结束日期": str(r.get("end_date")),
                "备注": r.get("notes", "") or "",
            }
            for r in existing
        ])
        if df.empty:
            df = pd.DataFrame(columns=["类型", "人员姓名", "角色", "开始日期", "结束日期", "备注"])

        with st.form("calendar_direct_edit_form", clear_on_submit=False):
            edited = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="calendar_direct_edit_editor",
                column_config={
                    "类型": st.column_config.SelectboxColumn(options=["内部稽查员", "兼职"]),
                    "角色": st.column_config.SelectboxColumn(options=["组长", "成员"]),
                },
            )
            c1, c2 = st.columns(2)
            save_direct_calendar = c1.form_submit_button("保存修改")
            sync_direct_calendar = c2.form_submit_button("同步到排班", type="primary")

        if save_direct_calendar or sync_direct_calendar:
            with db_session() as db:
                name_to_id = {a.name: a.id for a in db.query(Auditor).all()}
            rows_to_save = []
            for _, r in pd.DataFrame(edited).iterrows():
                nm = str(r.get("人员姓名", "")).strip()
                if not nm:
                    continue
                rows_to_save.append(
                    {
                        "auditor_id": None if str(r.get("类型", "")) == "兼职" else name_to_id.get(nm),
                        "person_name": nm,
                        "is_part_time": str(r.get("类型", "")) == "兼职",
                        "role": "leader" if str(r.get("角色", "")) == "组长" else "member",
                        "start_date": safe_parse_date(r.get("开始日期")),
                        "end_date": safe_parse_date(r.get("结束日期")),
                        "notes": str(r.get("备注", "")).strip(),
                    }
                )
            replace_direct_assignments(int(dtask_id), rows_to_save)
            if save_direct_calendar:
                st.success("已保存已定项目人员明细")
                st.rerun()
            if sync_direct_calendar and dtask:
                ok, msg = sync_task_schedules_from_direct_assignments(dtask)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with db_session() as db:
        all_ics = build_ics_events(db)
        st.download_button("导出全部 ICS 日历", all_ics, file_name="wnrh_all.ics", key="dl_all_ics")
        if auditor_id:
            one_ics = build_ics_events(db, auditor_id=auditor_id)
            st.download_button("导出当前稽查员 ICS 日历", one_ics, file_name=f"wnrh_auditor_{auditor_id}.ics", key="dl_one_ics")

# -------------------- 账号管理 --------------------
# -------------------- 账号管理 --------------------
elif page == "账号管理":
    st.subheader("账号管理")
    current_user = st.session_state.get("login_user", "")
    is_admin = bool(st.session_state.get("is_admin", False))
    is_super_admin = bool(st.session_state.get("is_super_admin", False))

    st.subheader("我的密码")
    with st.form("change_my_password", clear_on_submit=True):
        old_pw = st.text_input("当前密码", type="password")
        new_pw = st.text_input("新密码（至少6位）", type="password")
        new_pw2 = st.text_input("确认新密码", type="password")
        if st.form_submit_button("修改我的密码", type="primary"):
            if not check_login(current_user, old_pw):
                st.error("当前密码不正确")
            elif new_pw != new_pw2:
                st.error("两次输入的新密码不一致")
            else:
                ok, msg = update_auth_password(current_user, new_pw)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.divider()
    if not is_admin:
        st.info("当前账号仅可修改自己的密码。新增登录人员、重置他人密码、配置可见板块仅管理员可操作。")
    else:
        st.subheader("新增登录人员")
        with st.form("create_user_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            new_username = c1.text_input("新账号")
            new_password = c2.text_input("初始密码（至少6位）", type="password")
            role = c3.selectbox("权限", ["普通用户", "管理员", "主管理员"])
            if st.form_submit_button("新增账号", type="primary"):
                ok, msg = create_auth_user(
                    new_username,
                    new_password,
                    is_admin=(role in ("管理员", "主管理员")),
                    is_super_admin=(role == "主管理员"),
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        st.subheader("现有登录账号")
        users = list_auth_users()
        if users:
            rows = []
            for u in users:
                role_cn = "普通用户"
                if int(u.get("is_super_admin", 0)) == 1:
                    role_cn = "主管理员"
                elif int(u.get("is_admin", 0)) == 1:
                    role_cn = "管理员"
                rows.append({"账号": u.get("username"), "权限": role_cn, "创建时间": u.get("created_at") or ""})
            show_table(rows, 260)
        else:
            st.info("暂无账号")

        st.subheader("重置其他人员密码")
        user_labels = [u["username"] for u in users]
        if user_labels:
            with st.form("reset_password_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                reset_user = c1.selectbox("选择账号", user_labels)
                reset_pw = c2.text_input("新密码（至少6位）", type="password")
                if st.form_submit_button("重置密码"):
                    ok, msg = update_auth_password(reset_user, reset_pw)
                    if ok:
                        st.success(f"{reset_user}：{msg}")
                    else:
                        st.error(msg)

            st.subheader("删除登录账号")
            deletable = [u for u in user_labels if u not in ("admin", current_user)]
            if deletable:
                with st.form("delete_user_form", clear_on_submit=True):
                    del_user = st.selectbox("选择要删除的账号", deletable)
                    confirm_text = st.text_input("输入 DELETE 确认删除")
                    if st.form_submit_button("删除账号"):
                        if confirm_text != "DELETE":
                            st.error("请输入 DELETE 以确认删除")
                        else:
                            ok, msg = delete_auth_user(del_user, current_user)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

        st.divider()
        if not is_super_admin:
            st.info("提示：只有【主管理员】可以配置普通账号的可见板块。")
        else:
            st.subheader("普通账号可见板块配置（主管理员）")
            st.caption("勾选后保存：普通账号侧边栏仅显示被勾选的功能。管理员/主管理员默认全功能，不受此限制。")

            normal_users = []
            for u in users:
                if int(u.get("is_admin", 0)) == 1:
                    continue
                normal_users.append(u.get("username"))

            if not normal_users:
                st.info("暂无普通账号")
            else:
                target_user = st.selectbox("选择普通账号", normal_users, key="perm_target_user")
                current_pages = get_user_allowed_pages(target_user)
                selected_pages = st.multiselect(
                    "可见板块（勾选）",
                    options=ALL_PAGES,
                    default=current_pages,
                    key="perm_pages_multiselect",
                )
                c1, _ = st.columns([1, 3])
                if c1.button("保存可见板块", type="primary", key="save_perm_btn"):
                    ok, msg = set_user_allowed_pages(target_user, selected_pages)
                    if ok:
                        st.success(msg)
                        if str(target_user).strip() == str(current_user).strip():
                            st.session_state["allowed_pages"] = get_user_allowed_pages(current_user)
                        st.rerun()
                    else:
                        st.error(msg)

# -------------------- 数据清理 --------------------
elif page == "数据清理":
    st.subheader("数据清理")
    st.warning("当前无数据时，可直接清空所有业务表。此操作不可恢复。")
    with st.form("cleanup_form"):
        confirm = st.text_input("输入 CLEAR 确认清空")
        submitted = st.form_submit_button("清空全部业务数据", type="primary")
    if submitted:
        if confirm != "CLEAR":
            st.error("请输入 CLEAR")
        else:
            with db_session() as db:
                db.query(Schedule).delete()
                db.query(Task).delete()
                db.query(Auditor).delete()
                db.query(CityDistance).delete()
                db.query(City).delete()
                if safe_commit(db, "清空业务数据"):
                    clear_runtime_caches_after_data_change()
                    st.success("已清空")
                    st.rerun()

else:
    st.info("请选择左侧功能导航。")
