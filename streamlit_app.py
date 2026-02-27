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

import streamlit as st
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db import Base, SessionLocal, engine, ensure_schema
from app.models import Auditor, Task, Schedule, CityDistance, City
from app.scheduler import (
    build_candidates,
    propose_team,
    compute_from_city,
    get_distance_km,
    team_objective,
)
from app.seed_distances import SEED_CITY_DISTANCES, CITY_COORDS


st.set_page_config(page_title="审计排班系统", layout="wide")

# -------------------- 初始化 --------------------
Base.metadata.create_all(bind=engine)
ensure_schema()


@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def parse_date(value: str) -> date:
    return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()


def safe_parse_date(value: str | None) -> Optional[date]:
    v = str(value or "").strip()
    if not v:
        return None
    try:
        return parse_date(v)
    except Exception:
        return None


def d2s(v: Optional[date]) -> str:
    return v.strftime("%Y-%m-%d") if v else ""


def show_table(rows: list[dict], height: int = 380):
    if not rows:
        st.info("暂无数据")
        return
    st.dataframe(rows, use_container_width=True, height=height)


def seed_city_distances_if_needed(db: Session):
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
        db.commit()
    except Exception:
        db.rollback()


SEED_CITIES = [(name, latlon[0], latlon[1]) for name, latlon in CITY_COORDS.items()]


def seed_cities_if_needed(db: Session):
    if db.query(City).count() > 0:
        return
    for name, lat, lon in SEED_CITIES:
        db.add(City(name=name, lat=float(lat), lon=float(lon)))
    try:
        db.commit()
    except Exception:
        db.rollback()


with db_session() as db:
    seed_city_distances_if_needed(db)
    seed_cities_if_needed(db)



# -------------------- 登录认证（数据库持久化） --------------------
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
                    created_at TEXT
                )
                """
            )
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
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for username, password in _bootstrap_seed_users().items():
            clean_user = str(username).strip()
            if not clean_user:
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO auth_users (username, password_hash, is_admin, created_at)
                    VALUES (:username, :password_hash, :is_admin, :created_at)
                    """
                ),
                {
                    "username": clean_user,
                    "password_hash": hash_password(str(password)),
                    "is_admin": 1 if clean_user == "admin" else 0,
                    "created_at": now,
                },
            )


def list_auth_users() -> list[dict]:
    ensure_auth_table()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT username, is_admin, created_at FROM auth_users ORDER BY is_admin DESC, username ASC"
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
                "SELECT username, password_hash, is_admin, created_at FROM auth_users WHERE username = :username"
            ),
            {"username": clean_user},
        ).mappings().first()
    return dict(row) if row else None


def create_auth_user(username: str, password: str, is_admin: bool = False) -> tuple[bool, str]:
    ensure_auth_table()
    clean_user = str(username or "").strip()
    if not clean_user:
        return False, "账号不能为空"
    if len(clean_user) < 3:
        return False, "账号至少 3 位"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", clean_user):
        return False, "账号仅支持字母、数字、下划线、点、短横线"
    if len(str(password or "")) < 6:
        return False, "密码至少 6 位"
    if get_auth_user(clean_user):
        return False, "该账号已存在"
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO auth_users (username, password_hash, is_admin, created_at)
                VALUES (:username, :password_hash, :is_admin, :created_at)
                """
            ),
            {
                "username": clean_user,
                "password_hash": hash_password(password),
                "is_admin": 1 if is_admin else 0,
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
        return False, "新密码至少 6 位"
    if not get_auth_user(clean_user):
        return False, "账号不存在"
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE auth_users SET password_hash = :password_hash WHERE username = :username"
            ),
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


bootstrap_auth_users_if_needed()


def render_login():
    st.title("审计排班系统")
    st.subheader("账号密码登录")
    st.caption("首次使用默认管理员：admin / admin123。登录后可在【账号管理】中新增人员、修改密码。")
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
            st.success("登录成功，正在进入系统…")
            st.rerun()
        else:
            st.error("账号或密码错误")
    st.stop()


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

if not st.session_state["logged_in"]:
    render_login()

# -------------------- 常量 --------------------
STATUS_MAP = {"在岗": "active", "请假": "leave", "冻结": "frozen"}
STATUS_MAP_REV = {v: k for k, v in STATUS_MAP.items()}
BOOL_TRUE = {"是", "Y", "y", "yes", "YES", "True", "true", "1", "是/yes"}


# -------------------- 工具函数：模板/导入 --------------------
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
            ws.column_dimensions[col_letter].width = max(12, min(30, len(str(h)) * 2))
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


# -------------------- 工具函数：ICS --------------------
def ics_escape(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def build_ics_events(db: Session, auditor_id: int | None = None):
    q = db.query(Schedule).order_by(Schedule.id.desc())
    if auditor_id:
        q = q.filter(Schedule.auditor_id == auditor_id)
    sch = q.all()
    events = []
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for s in sch:
        a = db.query(Auditor).filter(Auditor.id == s.auditor_id).first()
        t = db.query(Task).filter(Task.id == s.task_id).first()
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
        events.extend([
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now}",
            f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}",
            f"DTEND:{end_exclusive.strftime('%Y%m%dT%H%M%S')}",
            f"SUMMARY:{ics_escape(summary)}",
            f"DESCRIPTION:{ics_escape(desc)}",
            "END:VEVENT",
        ])
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


# -------------------- 工具函数：业务 --------------------
def assign_team_to_task(db: Session, task: Task, leader_id: int, member_ids: list[int]):
    start_date = task.start_date
    end_date = (task.end_date or (task.start_date + timedelta(days=max(1, int(task.required_days or 1)) - 1)))

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
            for leader in leader_pool:
                member_pool = [c for c in member_pool_all if c.auditor_id != leader.auditor_id]
                need_n = max(0, int(t.required_headcount or 1) - 1)
                from app.scheduler import TeamProposal
                if need_n == 0:
                    cand_team = TeamProposal(leader=leader, members=[], team_score=leader.score, notes="optimized-single")
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
                extras = member_pool[need_n : need_n + 6]
                for ex in extras:
                    trial_members = base_members[:-1] + [ex] if base_members else [ex]
                    cand_team2 = TeamProposal(
                        leader=leader,
                        members=trial_members,
                        team_score=leader.score + sum(m.score for m in trial_members) / max(1, len(trial_members)),
                        notes="optimized-swap",
                    )
                    obj2 = team_objective(cand_team2, auditor_lookup, avg_cases, report["batch_week_counts"])
                    if best_obj is None or obj2 < best_obj:
                        best_obj, best_team = obj2, cand_team2
            if best_team:
                team = best_team

        if not team:
            report["skipped"].append({"task_id": t.id, "project": t.project_name, "reason": "无可用团队"})
            continue

        leader_id = int(team.leader.auditor_id)
        member_ids = [int(m.auditor_id) for m in team.members]
        assign_team_to_task(db, t, leader_id, member_ids)
        for aid in [leader_id] + member_ids:
            report["batch_week_counts"][aid] = int(report["batch_week_counts"].get(aid, 0)) + 1
        db.commit()
        report["assigned"].append(
            {
                "task_id": t.id,
                "project": t.project_name,
                "leader": team.leader.auditor_name,
                "members": [m.auditor_name for m in team.members],
            }
        )
    return report


def load_day_marks():
    try:
        p = Path(__file__).resolve().parent / "app" / "holidays_cn.json"
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            items = obj.get("items", [])
            if isinstance(items, list):
                return items
    except Exception:
        pass
    return []


# -------------------- 侧边栏 --------------------
st.sidebar.title("审计排班系统")
st.sidebar.caption(f"当前用户：{st.session_state.get('login_user', '')}")
if st.sidebar.button("退出登录"):
    st.session_state["logged_in"] = False
    st.session_state["is_admin"] = False
    st.session_state.pop("login_user", None)
    st.rerun()
page = st.sidebar.radio(
    "功能导航",
    [
        "智能排班",
        "批量排班",
        "稽查员管理",
        "任务管理",
        "城市距离",
        "城市坐标",
        "模板导入",
        "日历视图",
        "账号管理",
    ],
)

st.sidebar.caption("纯 Streamlit 版本：已去除 iframe / 127.0.0.1 依赖，可直接分享给同事使用。")

# -------------------- 页面：智能排班 --------------------
if page == "智能排班":
    st.title("智能排班")
    st.caption("先按硬约束筛选，再按“距离优先 + 适度负荷均衡”评分推荐。")
    with db_session() as db:
        tasks = db.query(Task).order_by(Task.id.desc()).all()
        schedules_recent = db.query(Schedule).order_by(Schedule.id.desc()).limit(120).all()

    if not tasks:
        st.info("请先在【任务管理】中录入任务。")
    else:
        task_options = {f"#{t.id} {t.project_name}｜{t.site_city}｜{d2s(t.start_date)}｜{t.required_days}天｜{t.required_headcount}人": t.id for t in tasks}
        selected_label = st.selectbox("选择任务", list(task_options.keys()))
        selected_task_id = task_options[selected_label]
        col_a, col_b = st.columns([1, 3])
        if col_a.button("生成推荐", type="primary"):
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
                    "error": None if team else "无可用团队方案：检查负责人/专家A组/人数不足/指定人冲突/每周上限/缓冲日冲突等。",
                }
            st.rerun()

        rec = st.session_state.get("recommend_result")
        if rec and rec.get("task_id") == selected_task_id:
            with db_session() as db:
                task = db.query(Task).filter(Task.id == selected_task_id).first()
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
                            [f"{m.auditor_name}（{m.group_level}，{m.from_city}，{m.km:.0f}km，评分 {m.score}）" for m in team.members]
                        )
                    )
                else:
                    st.write("**组员：** 无")
                st.caption(f"{team.notes}｜团队评分 {team.team_score}")
                default_member_ids = ",".join([str(m.auditor_id) for m in team.members])
                member_ids_text = st.text_input("确认指派前，可手工调整组员ID（逗号分隔）", value=default_member_ids)
                if st.button("确认指派", type="primary"):
                    ids = [x for x in re.split(r"[，,\s]+", member_ids_text.strip()) if x.strip()]
                    member_ids = []
                    for x in ids:
                        try:
                            member_ids.append(int(x))
                        except Exception:
                            pass
                    with db_session() as db:
                        task = db.query(Task).filter(Task.id == selected_task_id).first()
                        assign_team_to_task(db, task, int(team.leader.auditor_id), member_ids)
                        db.commit()
                    st.success("已确认指派，并已自动更新 last_city / last_date。")
                    st.session_state.pop("recommend_result", None)
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
    rows = []
    for s in schedules_recent:
        rows.append(
            {
                "ID": s.id,
                "任务": f"#{s.task_id} {s.task.project_name if s.task else ''}",
                "人员": f"#{s.auditor_id} {s.auditor.name if s.auditor else ''} ({s.auditor.group_level if s.auditor else ''})",
                "角色": s.role,
                "时间": f"{d2s(s.start_date)} ~ {d2s(s.end_date)}",
                "路线": f"{s.travel_from_city} → {s.travel_to_city}",
                "km": round(float(s.distance_km or 0), 1),
            }
        )
    show_table(rows, 360)
    if schedules_recent:
        delete_sid = st.selectbox("删除排班记录（按ID）", [s.id for s in schedules_recent])
        if st.button("删除所选排班记录"):
            with db_session() as db:
                obj = db.query(Schedule).filter(Schedule.id == delete_sid).first()
                if obj:
                    db.delete(obj)
                    db.commit()
            st.success("已删除")
            st.rerun()

# -------------------- 页面：批量排班 --------------------
elif page == "批量排班":
    st.title("批量排班")
    st.caption("只会处理“未排过”的任务；按 need_expert 优先 > 人数多优先 > 开始日期早 排序。")
    c1, c2, c3 = st.columns([1, 1, 1])
    date_start = c1.date_input("开始日期", value=date.today())
    date_end = c2.date_input("结束日期", value=date.today() + timedelta(days=30))
    mode = c3.selectbox("模式", ["greedy", "optimized"], format_func=lambda x: "快速模式（优先效率）" if x == "greedy" else "优化模式（优先成本与均衡）")
    if st.button("开始批量排班", type="primary"):
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

# -------------------- 页面：稽查员管理 --------------------
elif page == "稽查员管理":
    st.title("稽查员管理")
    with st.form("auditor_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("姓名*")
        gender = c2.selectbox("性别", ["男", "女"])
        group_level = c3.selectbox("等级", ["A", "B", "C"], index=1)
        can_lead = c4.selectbox("可带队", ["否", "是"])

        c5, c6, c7, c8 = st.columns(4)
        base_city = c5.text_input("常驻城市*")
        max_weekly_tasks = c6.number_input("每周上限", min_value=0, value=1, step=1)
        status_cn = c7.selectbox("状态", ["在岗", "请假", "冻结"])
        monthly_cases = c8.number_input("本月已排院次", min_value=0, value=0, step=1)

        c9, c10, c11, c12 = st.columns(4)
        travel_days = c9.number_input("本月差旅天数", min_value=0, value=0, step=1)
        continuous_days = c10.number_input("连续工作天数", min_value=0, value=0, step=1)
        last_city = c11.text_input("上次结束城市")
        last_date_text = c12.text_input("上次结束日期（YYYY-MM-DD）")

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
                            last_task_end_date=safe_parse_date(last_date_text),
                        )
                    )
                    db.commit()
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
    show_table(rows)
    if auditors:
        delete_id = st.selectbox("删除稽查员（按ID）", [a.id for a in auditors], format_func=lambda x: f"{x} - {next(a.name for a in auditors if a.id == x)}")
        if st.button("删除所选稽查员"):
            with db_session() as db:
                obj = db.query(Auditor).filter(Auditor.id == delete_id).first()
                if obj:
                    db.delete(obj)
                    db.commit()
            st.success("已删除")
            st.rerun()

# -------------------- 页面：任务管理 --------------------
elif page == "任务管理":
    st.title("任务管理")
    with st.form("task_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        project_name = c1.text_input("项目名称*")
        customer_name = c2.text_input("客户/申办方")
        need_expert = c3.selectbox("需要A带队", ["否", "是"])

        c4, c5, c6, c7 = st.columns(4)
        required_headcount = c4.number_input("所需人数", min_value=1, value=1, step=1)
        required_days = c5.number_input("任务天数", min_value=1, value=1, step=1)
        required_gender = c6.selectbox("性别要求", ["不限", "男", "女"])
        site_city = c7.text_input("中心城市*")

        c8, c9, c10 = st.columns(3)
        specified = c8.text_input("硬指定人员（可空）")
        preferred = c9.text_input("软指定专家/老师（可空）")
        start_date = c10.date_input("开始日期", value=date.today())
        end_date_text = st.text_input("结束日期（可空，YYYY-MM-DD）")

        if st.form_submit_button("新增任务", type="primary"):
            if not project_name.strip() or not site_city.strip():
                st.error("项目名称、中心城市必填。")
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
                            end_date=safe_parse_date(end_date_text),
                        )
                    )
                    db.commit()
                st.success("已新增")
                st.rerun()

    with db_session() as db:
        tasks = db.query(Task).order_by(Task.id.desc()).all()
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
    show_table(rows)
    if tasks:
        delete_id = st.selectbox("删除任务（按ID）", [t.id for t in tasks], format_func=lambda x: f"{x} - {next(t.project_name for t in tasks if t.id == x)}")
        if st.button("删除所选任务"):
            with db_session() as db:
                obj = db.query(Task).filter(Task.id == delete_id).first()
                if obj:
                    db.delete(obj)
                    db.commit()
            st.success("已删除")
            st.rerun()

# -------------------- 页面：城市距离 --------------------
elif page == "城市距离":
    st.title("城市距离")
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
                    rec = db.query(CityDistance).filter(CityDistance.from_city == from_city.strip(), CityDistance.to_city == to_city.strip()).first()
                    if rec:
                        rec.km = float(km)
                    else:
                        db.add(CityDistance(from_city=from_city.strip(), to_city=to_city.strip(), km=float(km)))
                    db.commit()
                st.success("已保存")
                st.rerun()

    with db_session() as db:
        dists = db.query(CityDistance).order_by(CityDistance.id.desc()).limit(300).all()
    rows = [{"ID": d.id, "from": d.from_city, "to": d.to_city, "km": round(float(d.km or 0), 1)} for d in dists]
    show_table(rows)
    if dists:
        delete_id = st.selectbox("删除距离记录（按ID）", [d.id for d in dists])
        if st.button("删除所选距离记录"):
            with db_session() as db:
                obj = db.query(CityDistance).filter(CityDistance.id == delete_id).first()
                if obj:
                    db.delete(obj)
                    db.commit()
            st.success("已删除")
            st.rerun()

# -------------------- 页面：城市坐标 --------------------
elif page == "城市坐标":
    st.title("城市坐标")
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
                    rec = db.query(City).filter(City.name == name.strip()).first()
                    if rec:
                        rec.lat = float(lat)
                        rec.lon = float(lon)
                    else:
                        db.add(City(name=name.strip(), lat=float(lat), lon=float(lon)))
                    db.commit()
                st.success("已保存")
                st.rerun()

    csv_file = st.file_uploader("批量导入城市坐标 CSV", type=["csv"], key="city_csv")
    if st.button("执行 CSV 导入"):
        if not csv_file:
            st.warning("请先上传 CSV 文件。")
        else:
            text = csv_file.getvalue().decode("utf-8-sig", errors="ignore")
            reader = csv.reader(io.StringIO(text))
            imported = 0
            with db_session() as db:
                for r in reader:
                    if not r or len(r) < 3:
                        continue
                    if str(r[0]).strip() in ("name", "城市", "city"):
                        continue
                    nm = str(r[0]).strip()
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
                db.commit()
            st.success(f"已导入 / 更新 {imported} 条城市坐标。")
            st.rerun()

    with db_session() as db:
        cities = db.query(City).order_by(City.id.desc()).limit(300).all()
    rows = [{"ID": c.id, "城市": c.name, "lat": round(float(c.lat), 6), "lon": round(float(c.lon), 6)} for c in cities]
    show_table(rows)
    if cities:
        delete_id = st.selectbox("删除城市（按ID）", [c.id for c in cities])
        if st.button("删除所选城市"):
            with db_session() as db:
                obj = db.query(City).filter(City.id == delete_id).first()
                if obj:
                    db.delete(obj)
                    db.commit()
            st.success("已删除")
            st.rerun()

# -------------------- 页面：模板导入 --------------------
elif page == "模板导入":
    st.title("模板导入")
    st.caption("下载模板 → 填写 → 上传导入，支持新增/更新。")

    # 下载：稽查员模板
    headers_a = [
        "姓名",
        "性别(男/女/不限)",
        "等级(A/B/C)",
        "可带队(是/否)",
        "常驻城市",
        "每周上限(院次)",
        "状态(在岗/请假/冻结)",
        "本月已排院次",
        "本月差旅天数",
        "连续工作天数",
        "上次结束城市",
        "上次结束日期(YYYY-MM-DD)",
    ]
    explain_a = ["必填", "可选", "必填", "可选", "必填", "默认1", "默认在岗", "默认0", "默认0", "默认0", "可空", "可空"]
    example_a = [
        ["张三", "男", "A", "是", "北京", 1, "在岗", 0, 0, 0, "苏州", "2026-01-20"],
        ["李四", "女", "B", "是", "上海", 2, "在岗", 0, 0, 0, "", ""],
    ]
    bio_a = make_xlsx_template(headers_a, [explain_a] + example_a, sheet_name="稽查员")
    st.download_button("下载稽查员模板（XLSX）", bio_a.getvalue(), file_name="稽查员模板.xlsx")

    headers_t = [
        "项目名称",
        "客户/申办方",
        "需要A带队(是/否)",
        "所需人数",
        "任务天数(用于推算结束)",
        "性别要求(男/女/不限)",
        "硬指定人员(可空)",
        "软指定专家/老师(可空)",
        "中心城市",
        "开始日期(YYYY-MM-DD)",
        "结束日期(可空)",
    ]
    explain_t = ["必填", "可空", "默认否", "默认1", "默认1", "默认不限", "可空", "可空(加分优先)", "必填", "必填", "可空(优先使用)"]
    example_t = [
        ["项目A", "申办方X", "否", 2, 2, "不限", "", "张三", "苏州", "2026-02-01", ""],
        ["项目B", "申办方Y", "是", 1, 3, "女", "", "", "北京", "2026-02-03", "2026-02-06"],
    ]
    bio_t = make_xlsx_template(headers_t, [explain_t] + example_t, sheet_name="任务")
    st.download_button("下载任务模板（XLSX）", bio_t.getvalue(), file_name="任务模板.xlsx")

    st.divider()
    auditor_xlsx = st.file_uploader("上传稽查员模板", type=["xlsx"], key="auditor_xlsx")
    if st.button("导入稽查员模板"):
        if not auditor_xlsx:
            st.warning("请先上传稽查员模板。")
        else:
            headers, rows = read_xlsx_rows(auditor_xlsx)
            if not rows:
                st.error("未读取到数据行。")
            else:
                aliases = {
                    "name": ["姓名", "name"],
                    "gender": ["性别(男/女/不限)", "gender"],
                    "group_level": ["等级(A/B/C)", "group_level"],
                    "can_lead_team": ["可带队(是/否)", "can_lead_team"],
                    "base_city": ["常驻城市", "base_city"],
                    "max_weekly_tasks": ["每周上限(院次)", "max_weekly_tasks"],
                    "status": ["状态(在岗/请假/冻结)", "status"],
                    "monthly_cases": ["本月已排院次", "monthly_cases"],
                    "travel_days": ["本月差旅天数", "travel_days"],
                    "continuous_days": ["连续工作天数", "continuous_days"],
                    "last_task_end_city": ["上次结束城市", "last_task_end_city"],
                    "last_task_end_date": ["上次结束日期(YYYY-MM-DD)", "last_task_end_date"],
                }
                imported = 0
                with db_session() as db:
                    for r in rows:
                        def gv(key, default=""):
                            i = find_idx(headers, aliases[key])
                            return default if i is None else r[i]

                        name = str(gv("name", "") or "").strip()
                        if not name:
                            continue
                        gender = str(gv("gender", "男") or "男").strip() or "男"
                        group_level = str(gv("group_level", "B") or "B").strip().upper() or "B"
                        can_lead_team = str(gv("can_lead_team", "否") or "否").strip()
                        base_city = str(gv("base_city", "") or "").strip()
                        max_weekly_tasks = int(gv("max_weekly_tasks", 1) or 1)
                        status = str(gv("status", "在岗") or "在岗").strip()
                        monthly_cases = int(gv("monthly_cases", 0) or 0)
                        travel_days = int(gv("travel_days", 0) or 0)
                        continuous_days = int(gv("continuous_days", 0) or 0)
                        last_city = str(gv("last_task_end_city", "") or "").strip() or None
                        last_date_raw = gv("last_task_end_date", "")
                        if isinstance(last_date_raw, datetime):
                            last_date = last_date_raw.date()
                        elif isinstance(last_date_raw, date):
                            last_date = last_date_raw
                        else:
                            last_date = safe_parse_date(str(last_date_raw or ""))

                        rec = db.query(Auditor).filter(Auditor.name == name).first()
                        if rec:
                            rec.gender = gender
                            rec.group_level = group_level
                            rec.can_lead_team = can_lead_team in BOOL_TRUE
                            rec.base_city = base_city
                            rec.max_weekly_tasks = max_weekly_tasks
                            rec.status = STATUS_MAP.get(status, status if status in STATUS_MAP_REV else "active")
                            rec.monthly_cases = monthly_cases
                            rec.travel_days = travel_days
                            rec.continuous_days = continuous_days
                            rec.last_task_end_city = last_city
                            rec.last_task_end_date = last_date
                        else:
                            db.add(
                                Auditor(
                                    name=name,
                                    gender=gender,
                                    group_level=group_level,
                                    can_lead_team=can_lead_team in BOOL_TRUE,
                                    base_city=base_city,
                                    max_weekly_tasks=max_weekly_tasks,
                                    status=STATUS_MAP.get(status, status if status in STATUS_MAP_REV else "active"),
                                    monthly_cases=monthly_cases,
                                    travel_days=travel_days,
                                    continuous_days=continuous_days,
                                    last_task_end_city=last_city,
                                    last_task_end_date=last_date,
                                )
                            )
                        imported += 1
                    db.commit()
                st.success(f"已导入 / 更新 {imported} 条稽查员记录。")
                st.rerun()

    st.divider()
    task_xlsx = st.file_uploader("上传任务模板", type=["xlsx"], key="task_xlsx")
    if st.button("导入任务模板"):
        if not task_xlsx:
            st.warning("请先上传任务模板。")
        else:
            headers, rows = read_xlsx_rows(task_xlsx)
            if not rows:
                st.error("未读取到数据行。")
            else:
                aliases = {
                    "project_name": ["项目名称", "project_name"],
                    "customer_name": ["客户/申办方", "customer_name"],
                    "need_expert": ["需要A带队(是/否)", "need_expert"],
                    "required_headcount": ["所需人数", "required_headcount"],
                    "required_days": ["任务天数(用于推算结束)", "required_days"],
                    "required_gender": ["性别要求(男/女/不限)", "required_gender"],
                    "specified_auditors": ["硬指定人员(可空)", "specified_auditors"],
                    "preferred_experts": ["软指定专家/老师(可空)", "preferred_experts"],
                    "site_city": ["中心城市", "site_city"],
                    "start_date": ["开始日期(YYYY-MM-DD)", "start_date"],
                    "end_date": ["结束日期(可空)", "end_date"],
                }
                imported = 0
                with db_session() as db:
                    for r in rows:
                        def gv(key, default=""):
                            i = find_idx(headers, aliases[key])
                            return default if i is None else r[i]

                        project_name = str(gv("project_name", "") or "").strip()
                        if not project_name:
                            continue
                        customer_name = str(gv("customer_name", "") or "").strip() or None
                        need_expert = str(gv("need_expert", "否") or "否").strip() in BOOL_TRUE
                        required_headcount = int(gv("required_headcount", 1) or 1)
                        required_days = int(gv("required_days", 1) or 1)
                        required_gender = str(gv("required_gender", "不限") or "不限").strip() or "不限"
                        specified = str(gv("specified_auditors", "") or "").strip() or None
                        preferred = str(gv("preferred_experts", "") or "").strip() or None
                        site_city = str(gv("site_city", "") or "").strip()
                        sd_raw = gv("start_date", "")
                        if isinstance(sd_raw, datetime):
                            start_d = sd_raw.date()
                        elif isinstance(sd_raw, date):
                            start_d = sd_raw
                        else:
                            start_d = safe_parse_date(str(sd_raw or ""))
                        if not start_d:
                            continue
                        ed_raw = gv("end_date", "")
                        if isinstance(ed_raw, datetime):
                            end_d = ed_raw.date()
                        elif isinstance(ed_raw, date):
                            end_d = ed_raw
                        else:
                            end_d = safe_parse_date(str(ed_raw or ""))

                        rec = db.query(Task).filter(Task.project_name == project_name, Task.start_date == start_d, Task.site_city == site_city).first()
                        if rec:
                            rec.customer_name = customer_name
                            rec.need_expert = need_expert
                            rec.required_headcount = required_headcount
                            rec.required_days = required_days
                            rec.required_gender = required_gender
                            rec.specified_auditors = specified
                            rec.preferred_experts = preferred
                            rec.end_date = end_d
                        else:
                            db.add(
                                Task(
                                    project_name=project_name,
                                    customer_name=customer_name,
                                    need_expert=need_expert,
                                    required_headcount=required_headcount,
                                    required_days=required_days,
                                    required_gender=required_gender,
                                    specified_auditors=specified,
                                    preferred_experts=preferred,
                                    site_city=site_city,
                                    start_date=start_d,
                                    end_date=end_d,
                                )
                            )
                        imported += 1
                    db.commit()
                st.success(f"已导入 / 更新 {imported} 条任务记录。")
                st.rerun()

# -------------------- 页面：账号管理 --------------------
elif page == "账号管理":
    st.title("账号管理")
    current_user = st.session_state.get("login_user", "")
    is_admin = bool(st.session_state.get("is_admin", False))

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
        st.info("当前账号仅可修改自己的密码。新增登录人员、重置他人密码仅管理员可操作。")
    else:
        st.subheader("新增登录人员")
        with st.form("create_user_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            new_username = c1.text_input("新账号")
            new_password = c2.text_input("初始密码（至少6位）", type="password")
            new_is_admin = c3.selectbox("权限", ["普通用户", "管理员"])
            if st.form_submit_button("新增账号", type="primary"):
                ok, msg = create_auth_user(new_username, new_password, is_admin=(new_is_admin == "管理员"))
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.subheader("现有登录账号")
        users = list_auth_users()
        if users:
            rows = []
            for u in users:
                rows.append(
                    {
                        "账号": u.get("username"),
                        "权限": "管理员" if int(u.get("is_admin", 0)) == 1 else "普通用户",
                        "创建时间": u.get("created_at") or "",
                    }
                )
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
                            else:
                                st.error(msg)
            else:
                st.info("当前没有可删除的账号（默认 admin 和当前登录账号不可删除）。")

# -------------------- 页面：日历视图 --------------------
elif page == "日历视图":
    st.title("日历视图")
    st.caption("按月查看排班、节假日标识，并支持导出 ICS 日历。")
    with db_session() as db:
        auditors = db.query(Auditor).order_by(Auditor.name.asc()).all()
        all_schedules = db.query(Schedule).order_by(Schedule.start_date.asc()).all()

    auditor_options = {"全部稽查员": None}
    for a in auditors:
        auditor_options[f"#{a.id} {a.name}"] = a.id
    c1, c2, c3 = st.columns(3)
    auditor_label = c1.selectbox("筛选稽查员", list(auditor_options.keys()))
    year = c2.selectbox("年份", list(range(date.today().year - 2, date.today().year + 3)), index=2)
    month = c3.selectbox("月份", list(range(1, 13)), index=date.today().month - 1)
    auditor_id = auditor_options[auditor_label]

    month_start = date(int(year), int(month), 1)
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    month_end = next_month - timedelta(days=1)

    filtered = []
    for s in all_schedules:
        if auditor_id and s.auditor_id != auditor_id:
            continue
        if s.start_date <= month_end and s.end_date >= month_start:
            filtered.append(s)

    day_marks = {it.get("date"): it for it in load_day_marks() if it.get("date", "")[:7] == month_start.strftime("%Y-%m")}

    st.subheader(f"{year}年{month}月 日历总览")
    weeks = []
    # Monday-first calendar matrix
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
            for s in filtered:
                if s.start_date <= day <= s.end_date:
                    proj = s.task.project_name if s.task else f"任务#{s.task_id}"
                    person = s.auditor.name if s.auditor else f"稽查员#{s.auditor_id}"
                    evs.append(f"{proj}｜{person}")
            color = "#ffffff"
            if day.month != month:
                color = "#f7f7f7"
            elif evs:
                color = "#eef6ff"
            cols[idx].markdown(
                f"<div style='border:1px solid #ddd;border-radius:8px;padding:8px;min-height:120px;background:{color};'>"
                f"<div style='font-weight:600'>{day.day}</div>"
                + (f"<div style='color:#d97706;font-size:12px'>{' / '.join(marks)}</div>" if marks else "")
                + ("" if not evs else "".join([f"<div style='font-size:12px;margin-top:4px'>{e}</div>" for e in evs[:4]]))
                + (f"<div style='font-size:12px;color:#666'>还有 {len(evs)-4} 项</div>" if len(evs) > 4 else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("本月排班明细")
    rows = []
    for s in filtered:
        rows.append(
            {
                "ID": s.id,
                "项目": s.task.project_name if s.task else "",
                "城市": s.task.site_city if s.task else "",
                "角色": "组长" if s.role == "leader" else "成员",
                "稽查员": s.auditor.name if s.auditor else "",
                "时间": f"{d2s(s.start_date)} ~ {d2s(s.end_date)}",
                "路线": f"{s.travel_from_city} → {s.travel_to_city}",
                "距离(km)": round(float(s.distance_km or 0), 1),
            }
        )
    show_table(rows, 320)

    with db_session() as db:
        all_ics = build_ics_events(db)
        st.download_button("导出全部 ICS 日历", all_ics, file_name="wnrh_all.ics")
        if auditor_id:
            one_ics = build_ics_events(db, auditor_id=auditor_id)
            st.download_button("导出当前稽查员 ICS 日历", one_ics, file_name=f"wnrh_auditor_{auditor_id}.ics")
