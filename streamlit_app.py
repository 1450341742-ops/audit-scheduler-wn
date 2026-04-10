
from __future__ import annotations

import calendar
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from app.db import Base, SessionLocal, engine, ensure_schema
from app.models import Auditor, Task, Schedule, CityDistance, City
from app.scheduler import build_candidates, propose_team, compute_from_city, get_distance_km

try:
    from app.seed_distances import SEED_CITY_DISTANCES, CITY_COORDS
except Exception:
    SEED_CITY_DISTANCES = []
    CITY_COORDS = {}

APP_NAME = "万宁睿和稽查排班"
st.set_page_config(page_title=APP_NAME, layout="wide")

STATUS_MAP = {"在岗": "active", "请假": "leave", "冻结": "frozen"}
STATUS_MAP_REV = {v: k for k, v in STATUS_MAP.items()}
DISEASE_PRESETS = [
    "内分泌","核药","CAR-T","慢性阻塞性肺疾病","血管性痴呆","结肠炎","哮喘","乳腺癌","皮肤病",
    "乙型肝炎","实体瘤","结核","骨质疏松症","肺癌","帕金森","失眠症","CIPD","特应性皮炎","白血病"
]
PHASE_PRESETS = ["IIb期","Ⅲ期","II期","I期","I 期","II/III期","I/IIa 期","II 期"]
CAPITAL_PRESETS = ["内资", "外资", "其他"]

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
.small-muted{color:#666;font-size:12px}
.calendar-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}
.calendar-cell{border:1px solid #e5e7eb;border-radius:10px;padding:8px;min-height:120px;background:#fff}
.calendar-day{font-weight:700;margin-bottom:6px}
.calendar-item{font-size:12px;background:#f7f7f7;border-radius:6px;padding:4px 6px;margin:4px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
</style>
""", unsafe_allow_html=True)

@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def safe_commit(db, context: str = "") -> bool:
    try:
        db.commit()
        return True
    except IntegrityError as e:
        db.rollback()
        st.error(f"数据库写入失败：{context}")
        st.exception(e)
        return False
    except Exception as e:
        db.rollback()
        st.error(f"数据库写入失败：{context}")
        st.exception(e)
        return False

def normalize_text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()

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
    s = str(value).strip()
    if not s:
        return None
    if " " in s:
        s = s.split(" ")[0]
    s = s.replace("/", "-").replace(".", "-")
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y%m%d"):
        try:
            d = datetime.strptime(s, fmt).date()
            return d if fmt != "%Y-%m" else d.replace(day=1)
        except Exception:
            pass
    return None

def d2s(v: Optional[date]) -> str:
    return v.strftime("%Y-%m-%d") if v else ""

def parse_names(raw: str):
    s = normalize_text(raw)
    if not s:
        return []
    for sep in ["，","、",";","；","|","/"]:
        s = s.replace(sep, ",")
    return [x.strip() for x in s.split(",") if x.strip()]

def task_end(task):
    ed = getattr(task, "end_date", None)
    if ed:
        return ed
    sd = getattr(task, "start_date")
    days = max(1, int(getattr(task, "required_days", 1) or 1))
    return sd + timedelta(days=days-1)

def overlap(a1, a2, b1, b2):
    return not (a2 < b1 or b2 < a1)

def seed_city_data():
    with db_session() as db:
        for name, coord in dict(CITY_COORDS).items():
            if db.query(City).filter(City.name == str(name).strip()).first():
                continue
            try:
                db.add(City(name=str(name).strip(), lat=float(coord[0]), lon=float(coord[1])))
                db.flush()
            except Exception:
                db.rollback()
        for a, b, km in SEED_CITY_DISTANCES:
            a = str(a).strip()
            b = str(b).strip()
            if not a or not b or a == b:
                continue
            if db.query(CityDistance).filter(CityDistance.from_city == a, CityDistance.to_city == b).first():
                continue
            try:
                db.add(CityDistance(from_city=a, to_city=b, km=float(km)))
                db.flush()
            except Exception:
                db.rollback()
        try:
            db.commit()
        except Exception:
            db.rollback()

@st.cache_resource(show_spinner=False)
def initialize_app():
    Base.metadata.create_all(bind=engine)
    try:
        ensure_schema()
    except Exception:
        pass
    seed_city_data()
    return True

initialize_app()

def task_meta_store():
    return st.session_state.setdefault("_task_meta", {})

def get_task_meta(task_id: int) -> dict:
    return task_meta_store().get(int(task_id), {
        "capital_type": "",
        "project_phase": "",
        "project_phase_other": "",
        "disease_area": "",
        "disease_area_other": "",
    })

def save_task_meta(task_id: int, capital_type: str, project_phase: str, project_phase_other: str, disease_area: str, disease_area_other: str):
    store = task_meta_store()
    store[int(task_id)] = {
        "capital_type": normalize_text(capital_type),
        "project_phase": normalize_text(project_phase),
        "project_phase_other": normalize_text(project_phase_other),
        "disease_area": normalize_text(disease_area),
        "disease_area_other": normalize_text(disease_area_other),
    }

def get_task_meta_display(meta: dict):
    phase = meta.get("project_phase_other") or meta.get("project_phase") or ""
    disease = meta.get("disease_area_other") or meta.get("disease_area") or ""
    capital = meta.get("capital_type") or ""
    return capital, phase, disease

def get_target_store():
    return st.session_state.setdefault("_targets", {})

def get_target(period_type: str, year: int, period_value: int) -> int:
    return int(get_target_store().get((period_type, int(year), int(period_value)), 0))

def save_target(period_type: str, year: int, period_value: int, value: int):
    get_target_store()[(period_type, int(year), int(period_value))] = int(value or 0)

def get_period_range(period_type: str, year: int, period_value: int):
    if period_type == "周":
        jan1 = date(year, 1, 1)
        start = jan1 + timedelta(days=(int(period_value)-1)*7)
        end = start + timedelta(days=6)
        return start, end
    if period_type == "月":
        start = date(year, int(period_value), 1)
        last_day = calendar.monthrange(year, int(period_value))[1]
        return start, date(year, int(period_value), last_day)
    if period_type == "季度":
        month_start = (int(period_value)-1)*3 + 1
        start = date(year, month_start, 1)
        month_end = month_start + 2
        last_day = calendar.monthrange(year, month_end)[1]
        return start, date(year, month_end, last_day)
    return date(year, 1, 1), date(year, 12, 31)

def get_subperiods(period_type: str, year: int, period_value: int):
    if period_type == "年":
        return [("月", m, f"{m}月") for m in range(1,13)]
    if period_type == "季度":
        month_start = (int(period_value)-1)*3 + 1
        return [("月", m, f"{m}月") for m in range(month_start, month_start+3)]
    if period_type == "月":
        start, end = get_period_range("月", year, period_value)
        weeks = []
        cur = start
        i = 1
        while cur <= end:
            weeks.append((None, i, f"第{i}周", cur, min(cur+timedelta(days=6), end)))
            cur += timedelta(days=7)
            i += 1
        return weeks
    return []

def query_base_data():
    with db_session() as db:
        auditors = db.query(Auditor).order_by(Auditor.id.desc()).all()
        tasks = db.query(Task).order_by(Task.start_date.desc(), Task.id.desc()).all()
        schedules = db.query(Schedule).options(joinedload(Schedule.task), joinedload(Schedule.auditor)).order_by(Schedule.start_date.desc(), Schedule.id.desc()).all()
    return auditors, tasks, schedules

def refresh():
    st.cache_data.clear()
    st.rerun()

def create_auditor(name, gender, group_level, can_lead_team, base_city, max_weekly_tasks, status_cn, monthly_cases):
    with db_session() as db:
        obj = Auditor(
            name=normalize_text(name),
            gender=normalize_text(gender) or "女",
            group_level=normalize_text(group_level) or "B",
            can_lead_team=bool(can_lead_team),
            base_city=normalize_text(base_city),
            max_weekly_tasks=max(1, int(max_weekly_tasks or 1)),
            status=STATUS_MAP.get(status_cn, "active"),
            monthly_cases=max(0, int(monthly_cases or 0)),
            travel_days=0,
            continuous_days=0,
            last_task_end_city=None,
            last_task_end_date=None,
        )
        db.add(obj)
        return safe_commit(db, "新增稽查员")

def update_auditor(obj_id, **kwargs):
    with db_session() as db:
        obj = db.query(Auditor).filter(Auditor.id == int(obj_id)).first()
        if not obj:
            return False
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return safe_commit(db, f"更新稽查员#{obj_id}")

def delete_auditor(obj_id):
    with db_session() as db:
        db.query(Schedule).filter(Schedule.auditor_id == int(obj_id)).delete()
        obj = db.query(Auditor).filter(Auditor.id == int(obj_id)).first()
        if obj:
            db.delete(obj)
        return safe_commit(db, f"删除稽查员#{obj_id}")

def create_task(project_name, customer_name, need_expert, required_headcount, required_days, required_gender, specified_auditors, preferred_experts, site_city, start_date, end_date):
    with db_session() as db:
        sd = safe_parse_date(start_date)
        ed = safe_parse_date(end_date) or (sd + timedelta(days=max(1, int(required_days or 1))-1))
        obj = Task(
            project_name=normalize_text(project_name),
            customer_name=normalize_text(customer_name) or None,
            need_expert=bool(need_expert),
            required_headcount=max(1, int(required_headcount or 1)),
            required_days=max(1, int(required_days or 1)),
            required_gender=normalize_text(required_gender) or "不限",
            specified_auditors=normalize_text(specified_auditors) or None,
            preferred_experts=normalize_text(preferred_experts) or None,
            site_city=normalize_text(site_city),
            start_date=sd,
            end_date=ed,
        )
        db.add(obj)
        ok = safe_commit(db, "新增任务")
        return ok, getattr(obj, "id", None)

def update_task(obj_id, **kwargs):
    with db_session() as db:
        obj = db.query(Task).filter(Task.id == int(obj_id)).first()
        if not obj:
            return False
        for k, v in kwargs.items():
            setattr(obj, k, v)
        schedules = db.query(Schedule).filter(Schedule.task_id == int(obj_id)).all()
        for s in schedules:
            s.start_date = obj.start_date
            s.end_date = obj.end_date
            s.travel_to_city = obj.site_city
        return safe_commit(db, f"更新任务#{obj_id}")

def delete_task(obj_id):
    with db_session() as db:
        db.query(Schedule).filter(Schedule.task_id == int(obj_id)).delete()
        obj = db.query(Task).filter(Task.id == int(obj_id)).first()
        if obj:
            db.delete(obj)
        return safe_commit(db, f"删除任务#{obj_id}")

def assign_team(task_id: int, leader_id: int, member_ids: list[int]):
    with db_session() as db:
        task = db.query(Task).filter(Task.id == int(task_id)).first()
        if not task:
            st.error("任务不存在")
            return False
        if db.query(Schedule).filter(Schedule.task_id == int(task_id)).count() > 0:
            st.warning("该任务已存在排班")
            return False
        start_date = task.start_date
        end_date = task.end_date or task_end(task)
        all_ids = [int(leader_id)] + [int(x) for x in member_ids if int(x) != int(leader_id)]
        for aid in all_ids:
            exist = db.query(Schedule).filter(Schedule.auditor_id == int(aid)).all()
            for s in exist:
                if overlap(start_date, end_date, s.start_date, s.end_date):
                    st.error(f"稽查员 {aid} 存在时间冲突")
                    return False
        for aid in all_ids:
            auditor = db.query(Auditor).filter(Auditor.id == int(aid)).first()
            if not auditor:
                continue
            from_city = compute_from_city(auditor, task)
            km = get_distance_km(db, from_city, task.site_city)
            db.add(Schedule(
                task_id=int(task.id),
                auditor_id=int(aid),
                role="leader" if int(aid)==int(leader_id) else "member",
                start_date=start_date,
                end_date=end_date,
                travel_from_city=from_city,
                travel_to_city=task.site_city,
                distance_km=float(km),
                score=0.0,
                status="confirmed",
            ))
            auditor.monthly_cases = int(auditor.monthly_cases or 0) + 1
            days = (end_date - start_date).days + 1
            auditor.travel_days = int(auditor.travel_days or 0) + max(0, days)
            auditor.continuous_days = max(int(auditor.continuous_days or 0), days)
            auditor.last_task_end_city = task.site_city
            auditor.last_task_end_date = end_date
        return safe_commit(db, f"任务排班#{task_id}")

def replace_manual_assignment(task_id: int, rows: list[dict]):
    with db_session() as db:
        task = db.query(Task).filter(Task.id == int(task_id)).first()
        if not task:
            return False
        db.query(Schedule).filter(Schedule.task_id == int(task_id)).delete()
        for r in rows:
            aid = int(r["auditor_id"])
            auditor = db.query(Auditor).filter(Auditor.id == aid).first()
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
                auditor_id=int(aid),
                role=normalize_text(r.get("role")) or "member",
                start_date=sd,
                end_date=ed,
                travel_from_city=from_city,
                travel_to_city=task.site_city,
                distance_km=float(km),
                score=0.0,
                status="confirmed",
            ))
        return safe_commit(db, f"手动排班#{task_id}")

def schedules_in_range(schedules, start_d: date, end_d: date):
    out = []
    for s in schedules:
        sd = getattr(s, "start_date", None)
        ed = getattr(s, "end_date", None) or sd
        if sd and overlap(start_d, end_d, sd, ed):
            out.append(s)
    return out

def tasks_in_range(tasks, start_d: date, end_d: date):
    out = []
    for t in tasks:
        sd = getattr(t, "start_date", None)
        ed = getattr(t, "end_date", None) or task_end(t)
        if sd and overlap(start_d, end_d, sd, ed):
            out.append(t)
    return out

def calc_progress(tasks, schedules, period_type: str, year: int, period_value: int):
    start_d, end_d = get_period_range(period_type, year, period_value)
    scheds = schedules_in_range(schedules, start_d, end_d)
    completed = len({int(s.task_id) for s in scheds})
    planned = len({int(t.id) for t in tasks_in_range(tasks, start_d, end_d)})
    target = get_target(period_type, year, period_value)
    rate = round((completed / target * 100), 1) if target > 0 else 0.0
    return {"start": start_d, "end": end_d, "completed": completed, "planned": planned, "target": target, "rate": rate, "scheds": scheds}

def build_trend_df(tasks, schedules, period_type: str, year: int, period_value: int):
    rows = []
    if period_type in ("年", "季度"):
        for ptype, pvalue, label in get_subperiods(period_type, year, period_value):
            info = calc_progress(tasks, schedules, ptype, year, pvalue)
            rows.append({"周期": label, "已完成院次": info["completed"], "目标院次": get_target(ptype, year, pvalue)})
    elif period_type == "月":
        for _, idx, label, sd, ed in get_subperiods(period_type, year, period_value):
            completed = len({int(s.task_id) for s in schedules_in_range(schedules, sd, ed)})
            rows.append({"周期": label, "已完成院次": completed})
    return pd.DataFrame(rows)

def month_range(year: int, month: int):
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last)

def build_calendar_event_map(tasks, schedules, year: int, month: int):
    start_d, end_d = month_range(year, month)
    event_map = {}
    task_by_id = {int(t.id): t for t in tasks}
    aud_by_id = {}
    for s in schedules:
        if getattr(s, "auditor", None):
            aud_by_id[int(s.auditor_id)] = s.auditor
    for s in schedules:
        sd = s.start_date
        ed = s.end_date or sd
        cur = max(sd, start_d)
        while cur <= min(ed, end_d):
            task = task_by_id.get(int(s.task_id))
            auditor = getattr(s, "auditor", None) or aud_by_id.get(int(s.auditor_id))
            txt = f"{'组长' if s.role=='leader' else '组员'} {task.project_name if task else s.task_id} | {auditor.name if auditor else s.auditor_id}"
            event_map.setdefault(cur, []).append(txt)
            cur += timedelta(days=1)
    return event_map

def make_calendar_png(year: int, month: int, event_map: dict):
    weeks = calendar.monthcalendar(year, month)
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")
    ax.set_title(f"{year}-{month:02d} 排班日历", fontsize=18, pad=20)
    cols = ["周一","周二","周三","周四","周五","周六","周日"]
    cell_text = []
    for wk in weeks:
        row = []
        for d in wk:
            if d == 0:
                row.append("")
            else:
                day = date(year, month, d)
                items = event_map.get(day, [])
                txt = str(d)
                for item in items[:2]:
                    txt += "\n• " + item[:18]
                if len(items) > 2:
                    txt += f"\n…另{len(items)-2}项"
                row.append(txt)
        cell_text.append(row)
    table = ax.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3.0)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def render_header(page_title: str, subtitle: str = ""):
    st.title(page_title)
    if subtitle:
        st.caption(subtitle)

auditors, tasks, schedules = query_base_data()
task_lookup = {int(t.id): t for t in tasks}
auditor_lookup = {int(a.id): a for a in auditors}

page = st.sidebar.radio("导航", ["经营看板", "智能排班", "任务管理", "稽查员管理", "指标统计", "日历视图"])

if page == "经营看板":
    render_header("经营看板", "基于现有 Auditor / Task / Schedule 生成，不依赖额外新表")
    today = date.today()
    month_info = calc_progress(tasks, schedules, "月", today.year, today.month)
    q = (today.month-1)//3 + 1
    quarter_info = calc_progress(tasks, schedules, "季度", today.year, q)
    year_info = calc_progress(tasks, schedules, "年", today.year, 1)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("本月目标院次", month_info["target"])
    c2.metric("本月已完成院次", month_info["completed"])
    c3.metric("本月完成率", f'{month_info["rate"]}%')
    c4.metric("当季已完成院次", quarter_info["completed"])
    c5.metric("当年已完成院次", year_info["completed"])
    # project structure
    project_rows = []
    for t in tasks:
        capital, phase, disease = get_task_meta_display(get_task_meta(int(t.id)))
        project_rows.append({"task_id": int(t.id), "项目名称": t.project_name, "内外资": capital or "未填", "分期": phase or "未填", "疾病领域": disease or "未填"})
    if project_rows:
        dfp = pd.DataFrame(project_rows)
        st.subheader("项目结构占比")
        col1, col2, col3 = st.columns(3)
        for col, name in [(col1, "内外资"), (col2, "分期"), (col3, "疾病领域")]:
            vc = dfp[name].value_counts().reset_index()
            vc.columns = [name, "项目数"]
            col.dataframe(vc, use_container_width=True, height=240)
    st.subheader("稽查员负荷")
    if auditors:
        eff_rows = []
        month_start, month_end = month_range(today.year, today.month)
        for a in auditors:
            sch = [s for s in schedules if int(s.auditor_id)==int(a.id) and overlap(month_start, month_end, s.start_date, s.end_date or s.start_date)]
            travel_days = 0
            task_ids = set()
            for s in sch:
                seg_start = max(month_start, s.start_date)
                seg_end = min(month_end, s.end_date or s.start_date)
                travel_days += (seg_end-seg_start).days + 1
                task_ids.add(int(s.task_id))
            total_days = (month_end-month_start).days + 1
            idle_days = max(0, total_days - travel_days)
            standard_max = int(a.monthly_cases or 6) if int(a.monthly_cases or 0) > 0 else 6
            standard_min = min(4, standard_max)
            completed_cases = len(task_ids)
            load_pct = round(completed_cases / max(1, standard_max) * 100, 1)
            overload = max(0.0, round((completed_cases - standard_max) / max(1, standard_max) * 100, 1))
            eff_rows.append({
                "稽查员": a.name,
                "月标准院次下限": standard_min,
                "月标准院次上限": standard_max,
                "本月已完成院次": completed_cases,
                "出差天数": travel_days,
                "空闲天数": idle_days,
                "负荷百分比": load_pct,
                "超负荷百分比": overload,
            })
        st.dataframe(pd.DataFrame(eff_rows), use_container_width=True, height=320)

elif page == "智能排班":
    render_header("智能排班", "选择未排班任务，系统自动推荐组长与组员")
    unscheduled_tasks = [t for t in tasks if all(int(s.task_id) != int(t.id) for s in schedules)]
    if not unscheduled_tasks:
        st.info("当前没有待排班任务")
    else:
        task_options = {f"{t.project_name} | {d2s(t.start_date)} | {t.site_city}": int(t.id) for t in unscheduled_tasks}
        chosen = st.selectbox("选择任务", list(task_options.keys()))
        task_id = task_options[chosen]
        task = task_lookup[int(task_id)]
        with db_session() as db:
            candidates = build_candidates(db, task, auditors, schedules)
            team = propose_team(task, candidates)
        if candidates:
            cand_df = pd.DataFrame([{
                "ID": c.auditor_id, "姓名": c.auditor_name, "组别": c.group_level, "可带队": c.can_lead_team,
                "出发地": c.from_city, "距离(km)": round(c.km,1), "分数": c.score, "说明": c.explain
            } for c in candidates])
            st.subheader("候选人")
            st.dataframe(cand_df, use_container_width=True, height=280)
        if team:
            st.success(f"推荐组长：{team.leader.auditor_name}；组员：{', '.join(m.auditor_name for m in team.members) if team.members else '无'}")
            if st.button("确认一键排班", type="primary"):
                ok = assign_team(task.id, team.leader.auditor_id, [m.auditor_id for m in team.members])
                if ok:
                    st.success("排班成功")
                    refresh()
        else:
            st.warning("未生成可用团队，请检查人员冲突、组长资质或任务要求")

elif page == "任务管理":
    render_header("任务管理", "新增 / 编辑任务，并可维护项目属性与手动排班")
    with st.expander("新增任务", expanded=False):
        with st.form("add_task_form"):
            c1, c2, c3, c4 = st.columns(4)
            project_name = c1.text_input("项目名称")
            customer_name = c2.text_input("客户名称")
            site_city = c3.text_input("中心城市")
            need_expert = c4.checkbox("需要A组长带队")
            d1, d2, d3, d4 = st.columns(4)
            required_headcount = d1.number_input("所需人数", 1, 10, 2)
            required_days = d2.number_input("所需天数", 1, 30, 1)
            required_gender = d3.selectbox("性别要求", ["不限","男","女"])
            start_d = d4.date_input("开始日期", value=date.today())
            e1, e2 = st.columns(2)
            specified = e1.text_input("硬指定人员")
            preferred = e2.text_input("优先人员")
            m1, m2, m3, m4, m5 = st.columns(5)
            capital_type = m1.selectbox("内资/外资", CAPITAL_PRESETS + [""])
            phase = m2.selectbox("项目分期", [""] + PHASE_PRESETS + ["其他"])
            phase_other = m3.text_input("分期补充")
            disease = m4.selectbox("疾病领域", [""] + DISEASE_PRESETS + ["其他"])
            disease_other = m5.text_input("疾病补充")
            submit = st.form_submit_button("新增任务", type="primary")
            if submit:
                end_d = start_d + timedelta(days=int(required_days)-1)
                ok, new_id = create_task(project_name, customer_name, need_expert, required_headcount, required_days, required_gender, specified, preferred, site_city, start_d, end_d)
                if ok:
                    if new_id:
                        save_task_meta(int(new_id), capital_type, phase, phase_other, disease, disease_other)
                    st.success("新增成功")
                    refresh()

    st.subheader("任务列表")
    for t in tasks:
        capital, phase_txt, disease_txt = get_task_meta_display(get_task_meta(int(t.id)))
        label = f"#{t.id} {t.project_name} | {d2s(t.start_date)} | {t.site_city}"
        with st.expander(label):
            c1, c2, c3 = st.columns(3)
            project_name = c1.text_input("项目名称", value=t.project_name, key=f"tpn_{t.id}")
            customer_name = c2.text_input("客户名称", value=t.customer_name or "", key=f"tcn_{t.id}")
            site_city = c3.text_input("中心城市", value=t.site_city or "", key=f"tct_{t.id}")
            d1, d2, d3, d4 = st.columns(4)
            need_expert = d1.checkbox("需要A组长", value=bool(t.need_expert), key=f"tex_{t.id}")
            required_headcount = d2.number_input("人数", 1, 10, int(t.required_headcount or 1), key=f"thc_{t.id}")
            required_days = d3.number_input("天数", 1, 30, int(t.required_days or 1), key=f"tdy_{t.id}")
            required_gender = d4.selectbox("性别", ["不限","男","女"], index=["不限","男","女"].index(t.required_gender or "不限") if (t.required_gender or "不限") in ["不限","男","女"] else 0, key=f"tgd_{t.id}")
            e1, e2, e3, e4 = st.columns(4)
            specified = e1.text_input("硬指定人员", value=t.specified_auditors or "", key=f"tsp_{t.id}")
            preferred = e2.text_input("优先人员", value=t.preferred_experts or "", key=f"tpr_{t.id}")
            sd = e3.date_input("开始日期", value=t.start_date, key=f"tsd_{t.id}")
            ed = e4.date_input("结束日期", value=t.end_date or task_end(t), key=f"ted_{t.id}")
            meta = get_task_meta(int(t.id))
            m1, m2, m3, m4, m5 = st.columns(5)
            capital_type = m1.selectbox("内资/外资", CAPITAL_PRESETS + [""], index=(CAPITAL_PRESETS + [""]).index(meta.get("capital_type","")) if meta.get("capital_type","") in (CAPITAL_PRESETS + [""]) else len(CAPITAL_PRESETS), key=f"cap_{t.id}")
            phase = m2.selectbox("项目分期", [""] + PHASE_PRESETS + ["其他"], index=([""] + PHASE_PRESETS + ["其他"]).index(meta.get("project_phase","")) if meta.get("project_phase","") in ([""] + PHASE_PRESETS + ["其他"]) else 0, key=f"pha_{t.id}")
            phase_other = m3.text_input("分期补充", value=meta.get("project_phase_other",""), key=f"pho_{t.id}")
            disease = m4.selectbox("疾病领域", [""] + DISEASE_PRESETS + ["其他"], index=([""] + DISEASE_PRESETS + ["其他"]).index(meta.get("disease_area","")) if meta.get("disease_area","") in ([""] + DISEASE_PRESETS + ["其他"]) else 0, key=f"dis_{t.id}")
            disease_other = m5.text_input("疾病补充", value=meta.get("disease_area_other",""), key=f"dio_{t.id}")
            b1, b2 = st.columns(2)
            if b1.button("保存任务", key=f"save_task_{t.id}"):
                ok = update_task(t.id,
                    project_name=normalize_text(project_name),
                    customer_name=normalize_text(customer_name) or None,
                    need_expert=bool(need_expert),
                    required_headcount=int(required_headcount),
                    required_days=int(required_days),
                    required_gender=normalize_text(required_gender),
                    specified_auditors=normalize_text(specified) or None,
                    preferred_experts=normalize_text(preferred) or None,
                    site_city=normalize_text(site_city),
                    start_date=safe_parse_date(sd),
                    end_date=safe_parse_date(ed),
                )
                if ok:
                    save_task_meta(int(t.id), capital_type, phase, phase_other, disease, disease_other)
                    st.success("保存成功")
                    refresh()
            if b2.button("删除任务", key=f"del_task_{t.id}"):
                if delete_task(t.id):
                    st.success("已删除")
                    refresh()
            st.markdown("**已定项目录入**")
            current = [s for s in schedules if int(s.task_id) == int(t.id)]
            rows = []
            for s in current:
                a = auditor_lookup.get(int(s.auditor_id))
                rows.append({
                    "项目名称": t.project_name,
                    "人员姓名": a.name if a else "",
                    "角色": "组长" if s.role == "leader" else "组员",
                    "开始日期": s.start_date,
                    "结束日期": s.end_date or s.start_date,
                })
            if not rows:
                rows = [{"项目名称": t.project_name, "人员姓名": "", "角色": "组员", "开始日期": t.start_date, "结束日期": t.end_date or task_end(t)}]
            edit_df = st.data_editor(pd.DataFrame(rows), num_rows="dynamic", key=f"dir_{t.id}", use_container_width=True)
            if st.button("保存已定项目录入", key=f"save_dir_{t.id}"):
                name_to_id = {a.name: int(a.id) for a in auditors}
                payload = []
                for _, r in edit_df.iterrows():
                    name = normalize_text(r.get("人员姓名"))
                    if not name or name not in name_to_id:
                        continue
                    payload.append({
                        "auditor_id": name_to_id[name],
                        "role": "leader" if normalize_text(r.get("角色")) == "组长" else "member",
                        "start_date": r.get("开始日期"),
                        "end_date": r.get("结束日期"),
                    })
                if replace_manual_assignment(t.id, payload):
                    st.success("已保存到排班表")
                    refresh()

elif page == "稽查员管理":
    render_header("稽查员管理", "维护人员基础资料与每月标准院次")
    with st.expander("新增稽查员", expanded=False):
        with st.form("add_auditor"):
            c1, c2, c3, c4 = st.columns(4)
            name = c1.text_input("姓名")
            gender = c2.selectbox("性别", ["女","男"])
            group_level = c3.selectbox("组别", ["A","B","C"])
            can_lead = c4.checkbox("可带队")
            d1, d2, d3, d4 = st.columns(4)
            base_city = d1.text_input("常驻城市")
            max_weekly = d2.number_input("每周最大任务数", 1, 10, 1)
            status_cn = d3.selectbox("状态", ["在岗","请假","冻结"])
            monthly_cases = d4.number_input("每月标准院次上限", 0, 20, 6)
            submit = st.form_submit_button("新增", type="primary")
            if submit:
                if create_auditor(name, gender, group_level, can_lead, base_city, max_weekly, status_cn, monthly_cases):
                    st.success("新增成功")
                    refresh()
    for a in auditors:
        with st.expander(f"#{a.id} {a.name} | {a.base_city} | {a.group_level}"):
            c1, c2, c3, c4 = st.columns(4)
            name = c1.text_input("姓名", value=a.name, key=f"an_{a.id}")
            gender = c2.selectbox("性别", ["女","男"], index=["女","男"].index(a.gender or "女") if (a.gender or "女") in ["女","男"] else 0, key=f"ag_{a.id}")
            group_level = c3.selectbox("组别", ["A","B","C"], index=["A","B","C"].index(a.group_level or "B") if (a.group_level or "B") in ["A","B","C"] else 1, key=f"al_{a.id}")
            can_lead = c4.checkbox("可带队", value=bool(a.can_lead_team), key=f"ac_{a.id}")
            d1, d2, d3, d4 = st.columns(4)
            base_city = d1.text_input("常驻城市", value=a.base_city or "", key=f"ab_{a.id}")
            max_weekly = d2.number_input("每周最大任务数", 1, 10, int(a.max_weekly_tasks or 1), key=f"am_{a.id}")
            status_cn = d3.selectbox("状态", ["在岗","请假","冻结"], index=["在岗","请假","冻结"].index(STATUS_MAP_REV.get(a.status, "在岗")), key=f"as_{a.id}")
            monthly_cases = d4.number_input("每月标准院次上限", 0, 20, int(a.monthly_cases or 6), key=f"aj_{a.id}")
            b1, b2 = st.columns(2)
            if b1.button("保存稽查员", key=f"save_a_{a.id}"):
                ok = update_auditor(a.id,
                    name=normalize_text(name),
                    gender=normalize_text(gender),
                    group_level=normalize_text(group_level),
                    can_lead_team=bool(can_lead),
                    base_city=normalize_text(base_city),
                    max_weekly_tasks=int(max_weekly),
                    status=STATUS_MAP.get(status_cn, "active"),
                    monthly_cases=int(monthly_cases),
                )
                if ok:
                    st.success("保存成功")
                    refresh()
            if b2.button("删除稽查员", key=f"del_a_{a.id}"):
                if delete_auditor(a.id):
                    st.success("已删除")
                    refresh()

elif page == "指标统计":
    render_header("指标统计", "按周 / 月 / 季 / 年维护院次目标，并自动计算完成率")
    y = st.number_input("年份", 2024, 2035, date.today().year)
    c1, c2, c3 = st.columns(3)
    period_type = c1.selectbox("统计周期", ["周","月","季度","年"])
    if period_type == "周":
        period_value = c2.number_input("第几周", 1, 53, 1)
    elif period_type == "月":
        period_value = c2.selectbox("月份", list(range(1,13)), index=date.today().month-1)
    elif period_type == "季度":
        period_value = c2.selectbox("季度", [1,2,3,4], index=((date.today().month-1)//3))
    else:
        period_value = 1
        c2.write("整年")
    target = c3.number_input("目标院次", 0, 9999, get_target(period_type, int(y), int(period_value)))
    if st.button("保存院次指标", type="primary"):
        save_target(period_type, int(y), int(period_value), int(target))
        st.success("已保存")
    info = calc_progress(tasks, schedules, period_type, int(y), int(period_value))
    m1, m2, m3 = st.columns(3)
    m1.metric("目标院次", info["target"])
    m2.metric("已完成院次", info["completed"])
    m3.metric("完成率", f'{info["rate"]}%')
    trend_df = build_trend_df(tasks, schedules, period_type, int(y), int(period_value))
    if not trend_df.empty:
        st.subheader("趋势图")
        chart_df = trend_df.set_index("周期")
        st.line_chart(chart_df)
        st.dataframe(trend_df, use_container_width=True)
    st.subheader("稽查员效率统计")
    start_d, end_d = get_period_range(period_type, int(y), int(period_value))
    eff_rows = []
    for a in auditors:
        sch = [s for s in schedules if int(s.auditor_id)==int(a.id) and overlap(start_d, end_d, s.start_date, s.end_date or s.start_date)]
        travel_days = 0
        task_ids = set()
        for s in sch:
            seg_start = max(start_d, s.start_date)
            seg_end = min(end_d, s.end_date or s.start_date)
            travel_days += (seg_end-seg_start).days + 1
            task_ids.add(int(s.task_id))
        total_days = (end_d-start_d).days + 1
        idle_days = max(0, total_days - travel_days)
        standard_max = int(a.monthly_cases or 6) if int(a.monthly_cases or 0) > 0 else 6
        completed_cases = len(task_ids)
        load_pct = round(completed_cases / max(1, standard_max) * 100, 1)
        overload = max(0.0, round((completed_cases - standard_max) / max(1, standard_max) * 100, 1))
        eff_rows.append({
            "稽查员": a.name,
            "标准月院次上限": standard_max,
            "已完成院次": completed_cases,
            "出差天数": travel_days,
            "空闲天数": idle_days,
            "负荷程度%": load_pct,
            "超负荷%": overload,
        })
    st.dataframe(pd.DataFrame(eff_rows), use_container_width=True, height=320)

elif page == "日历视图":
    render_header("日历视图", "单页月历，支持下载当月日历图片")
    c1, c2 = st.columns(2)
    y = c1.number_input("年份", 2024, 2035, date.today().year, key="cy")
    m = c2.selectbox("月份", list(range(1,13)), index=date.today().month-1, key="cm")
    event_map = build_calendar_event_map(tasks, schedules, int(y), int(m))
    weeks = calendar.monthcalendar(int(y), int(m))
    weekdays = ["周一","周二","周三","周四","周五","周六","周日"]
    cols = st.columns(7)
    for i, wd in enumerate(weekdays):
        cols[i].markdown(f"**{wd}**")
    for wk in weeks:
        cols = st.columns(7)
        for i, d in enumerate(wk):
            if d == 0:
                cols[i].markdown("<div class='calendar-cell'></div>", unsafe_allow_html=True)
            else:
                day = date(int(y), int(m), int(d))
                items = event_map.get(day, [])
                html = f"<div class='calendar-cell'><div class='calendar-day'>{d}</div>"
                for item in items[:2]:
                    html += f"<div class='calendar-item'>{item}</div>"
                if len(items) > 2:
                    html += f"<div class='calendar-item'>…另{len(items)-2}项</div>"
                html += "</div>"
                cols[i].markdown(html, unsafe_allow_html=True)
                if items:
                    with cols[i].expander("展开", expanded=False):
                        for item in items:
                            st.write(item)
    png = make_calendar_png(int(y), int(m), event_map)
    st.download_button("下载当月日历图片 PNG", data=png, file_name=f"calendar_{int(y)}_{int(m):02d}.png", mime="image/png")
