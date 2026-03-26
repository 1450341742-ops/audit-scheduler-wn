from __future__ import annotations

from datetime import timedelta
from math import radians, sin, cos, sqrt, atan2
from types import SimpleNamespace

from sqlalchemy.orm import Session

from app.models import Auditor, Task, Schedule, CityDistance, City


def _norm(v) -> str:
    return str(v or "").strip()


def _task_start(task: Task):
    return getattr(task, "start_date")


def _task_end(task: Task):
    end_date = getattr(task, "end_date", None)
    if end_date:
        return end_date
    start = _task_start(task)
    days = max(1, int(getattr(task, "required_days", 1) or 1))
    return start + timedelta(days=days - 1)


def _overlap(a_start, a_end, b_start, b_end) -> bool:
    return not (a_end < b_start or b_end < a_start)


def _parse_names(raw):
    s = _norm(raw)
    if not s:
        return []
    for sep in ["，", ";", "；", "/", "|"]:
        s = s.replace(sep, ",")
    return [x.strip() for x in s.split(",") if x.strip()]


def _same_week(d1, d2) -> bool:
    return d1.isocalendar()[:2] == d2.isocalendar()[:2]


def _count_week_tasks(auditor_id: int, task: Task, schedules_all):
    start = _task_start(task)
    count = 0
    for s in schedules_all:
        if int(getattr(s, "auditor_id")) != int(auditor_id):
            continue
        s_start = getattr(s, "start_date", None)
        if s_start and _same_week(s_start, start):
            count += 1
    return count


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    dlat = radians(float(lat2) - float(lat1))
    dlon = radians(float(lon2) - float(lon1))
    a = sin(dlat / 2) ** 2 + cos(radians(float(lat1))) * cos(radians(float(lat2))) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def compute_from_city(auditor: Auditor, task: Task) -> str:
    """
    优先使用上次结束城市；若没有则使用常驻城市。
    只要上次结束日期不晚于当前任务开始日期，就允许从上次结束城市出发。
    """
    last_city = _norm(getattr(auditor, "last_task_end_city", None))
    last_date = getattr(auditor, "last_task_end_date", None)
    base_city = _norm(getattr(auditor, "base_city", ""))

    if last_city and last_date and last_date <= _task_start(task):
        return last_city

    return last_city or base_city


def get_distance_km(db: Session, from_city: str, to_city: str) -> float:
    from_city = _norm(from_city)
    to_city = _norm(to_city)

    if not from_city or not to_city:
        return 9999.0
    if from_city == to_city:
        return 0.0

    rec = db.query(CityDistance).filter(
        CityDistance.from_city == from_city,
        CityDistance.to_city == to_city
    ).first()
    if rec and getattr(rec, "km", None) is not None:
        return float(rec.km)

    rec_rev = db.query(CityDistance).filter(
        CityDistance.from_city == to_city,
        CityDistance.to_city == from_city
    ).first()
    if rec_rev and getattr(rec_rev, "km", None) is not None:
        return float(rec_rev.km)

    c1 = db.query(City).filter(City.name == from_city).first()
    c2 = db.query(City).filter(City.name == to_city).first()
    if c1 and c2:
        km = float(_haversine_km(c1.lat, c1.lon, c2.lat, c2.lon))
        try:
            db.add(CityDistance(from_city=from_city, to_city=to_city, km=km))
            db.commit()
        except Exception:
            db.rollback()
        return km

    return 9999.0


def build_candidates(db: Session, task: Task, auditors, schedules_all):
    """
    稳定版候选池逻辑：
    - 这里只做基础条件筛选
    - 不因为 need_expert=True 就把所有候选人限制为 A
    - “A带队”只在 propose_team 阶段限制 leader
    """
    task_start = _task_start(task)
    task_end = _task_end(task)

    required_gender = _norm(getattr(task, "required_gender", "不限"))
    site_city = _norm(getattr(task, "site_city", ""))
    specified_names = set(_parse_names(getattr(task, "specified_auditors", None)))
    preferred_names = set(_parse_names(getattr(task, "preferred_experts", None)))

    candidates = []

    for auditor in auditors:
        auditor_id = int(getattr(auditor, "id"))
        name = _norm(getattr(auditor, "name", ""))
        status = _norm(getattr(auditor, "status", "active"))
        gender = _norm(getattr(auditor, "gender", ""))
        group_level = _norm(getattr(auditor, "group_level", ""))
        can_lead_team = bool(getattr(auditor, "can_lead_team", False))
        max_weekly_tasks = int(getattr(auditor, "max_weekly_tasks", 1) or 1)

        # 1) 状态
        if status != "active":
            continue

        # 2) 性别要求
        if required_gender in ("男", "女") and gender and gender != required_gender:
            continue

        # 3) 硬指定：如果填写了硬指定，则候选池只保留指定名单
        if specified_names and name not in specified_names:
            continue

        # 4) 时间冲突：已有排班时间重叠则过滤
        conflict = False
        for s in schedules_all:
            if int(getattr(s, "auditor_id")) != auditor_id:
                continue
            s_start = getattr(s, "start_date", None)
            s_end = getattr(s, "end_date", None) or s_start
            if s_start and s_end and _overlap(task_start, task_end, s_start, s_end):
                conflict = True
                break
        if conflict:
            continue

        # 5) 上次结束日期冲突
        last_date = getattr(auditor, "last_task_end_date", None)
        if last_date and last_date >= task_start:
            continue

        # 6) 周上限
        week_count = _count_week_tasks(auditor_id, task, schedules_all)
        if week_count >= max_weekly_tasks:
            continue

        from_city = compute_from_city(auditor, task)
        km = float(get_distance_km(db, from_city, site_city))

        # 评分：距离越近越高；A/B/带队/软指定有加分
        distance_penalty = min(km / 35.0, 60.0)
        level_bonus = {"A": 12.0, "B": 6.0, "C": 0.0}.get(group_level, 0.0)
        lead_bonus = 8.0 if can_lead_team else 0.0
        prefer_bonus = 10.0 if (preferred_names and name in preferred_names) else 0.0

        score = round(max(0.0, 100.0 - distance_penalty + level_bonus + lead_bonus + prefer_bonus), 1)

        explain_parts = [
            f"出发地:{from_city}",
            f"距离{round(km, 1)}km",
            f"组别{group_level}+{level_bonus}",
            f"带队+{lead_bonus}",
        ]
        if prefer_bonus:
            explain_parts.append(f"软指定+{prefer_bonus}")

        candidates.append(
            SimpleNamespace(
                auditor_id=auditor_id,
                auditor_name=name,
                group_level=group_level,
                can_lead_team=can_lead_team,
                from_city=from_city,
                km=km,
                score=score,
                explain="；".join(explain_parts),
            )
        )

    # 评分优先，其次距离近，其次ID小
    candidates.sort(key=lambda c: (-float(c.score), float(c.km), int(c.auditor_id)))
    return candidates


def propose_team(task: Task, candidates):
    """
    稳定版组队逻辑：
    - need_expert=False：组长只需可带队
    - need_expert=True：仅组长必须 A 且可带队
    - 组员从剩余全部可用候选人中补齐，不强制 A
    """
    if not candidates:
        return None

    need_n = max(1, int(getattr(task, "required_headcount", 1) or 1))
    need_members = max(0, need_n - 1)
    need_expert = bool(getattr(task, "need_expert", False))

    leader_pool = [c for c in candidates if bool(getattr(c, "can_lead_team", False))]
    if need_expert:
        leader_pool = [c for c in leader_pool if _norm(getattr(c, "group_level", "")) == "A"]

    if not leader_pool:
        return None

    best_team = None
    best_score = None

    for leader in leader_pool:
        member_pool = [c for c in candidates if int(getattr(c, "auditor_id")) != int(getattr(leader, "auditor_id"))]
        members = member_pool[:need_members]

        if len(members) < need_members:
            continue

        avg_member_score = (sum(float(getattr(m, "score", 0.0)) for m in members) / len(members)) if members else 0.0
        team_score = round(float(getattr(leader, "score", 0.0)) + avg_member_score, 1)

        notes = "最终稳定逻辑：仅组长要求A带队，组员按全部可用候选人补齐" if need_expert else "标准逻辑：组长可带队，组员按全部可用候选人补齐"

        team = SimpleNamespace(
            leader=leader,
            members=members,
            team_score=team_score,
            notes=notes,
        )

        if best_score is None or team_score > best_score:
            best_score = team_score
            best_team = team

    return best_team


def team_objective(team, auditor_lookup, avg_cases: float, batch_week_counts):
    """
    值越小越优：
    - 总距离越小越好
    - 当前负荷越接近平均越好
    - 批量过程中周内重复使用越少越好
    """
    ids = [int(getattr(team.leader, "auditor_id"))] + [int(getattr(m, "auditor_id")) for m in getattr(team, "members", [])]

    total_distance = float(getattr(team.leader, "km", 0.0)) + sum(float(getattr(m, "km", 0.0)) for m in getattr(team, "members", []))

    load_penalty = 0.0
    batch_penalty = 0.0

    for aid in ids:
        auditor = auditor_lookup.get(aid)
        current_cases = int(getattr(auditor, "monthly_cases", 0) or 0) if auditor else 0
        load_penalty += abs(current_cases - avg_cases)
        batch_penalty += float(batch_week_counts.get(aid, 0)) * 2.0

    return round(total_distance * 0.8 + load_penalty * 1.2 + batch_penalty, 3)
