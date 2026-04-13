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
    for sep in ["，", "、", ";", "；", "/", "|"]:
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
    稳定版候选池逻辑（保留原功能）：
    - 只做基础条件筛选
    - 不因为 need_expert=True 就把所有候选人限制为 A
    - “A带队”只在 propose_team 阶段限制 leader
    - 预先按 auditor_id 分组排班，避免每个候选人都全表扫描 schedules_all
    """
    task_start = _task_start(task)
    task_end = _task_end(task)

    required_gender = _norm(getattr(task, "required_gender", "不限"))
    site_city = _norm(getattr(task, "site_city", ""))
    preferred_names = set(_parse_names(getattr(task, "preferred_experts", None)))

    schedules_by_auditor = {}
    for s in schedules_all:
        aid = int(getattr(s, "auditor_id"))
        schedules_by_auditor.setdefault(aid, []).append(s)

    candidates = []

    for auditor in auditors:
        auditor_id = int(getattr(auditor, "id"))
        name = _norm(getattr(auditor, "name", ""))
        status = _norm(getattr(auditor, "status", "active"))
        gender = _norm(getattr(auditor, "gender", ""))
        group_level = _norm(getattr(auditor, "group_level", ""))
        can_lead_team = bool(getattr(auditor, "can_lead_team", False))
        max_weekly_tasks = int(getattr(auditor, "max_weekly_tasks", 1) or 1)

        if status != "active":
            continue
        if required_gender in ("男", "女") and gender and gender != required_gender:
            continue

        own_schedules = schedules_by_auditor.get(auditor_id, [])
        conflict = False
        for s in own_schedules:
            s_start = getattr(s, "start_date", None)
            s_end = getattr(s, "end_date", None) or s_start
            if s_start and s_end and _overlap(task_start, task_end, s_start, s_end):
                conflict = True
                break
        if conflict:
            continue

        last_date = getattr(auditor, "last_task_end_date", None)
        if last_date and last_date >= task_start:
            continue

        week_count = _count_week_tasks(auditor_id, task, own_schedules)
        if week_count >= max_weekly_tasks:
            continue

        from_city = compute_from_city(auditor, task)
        km = float(get_distance_km(db, from_city, site_city))

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

    candidates.sort(key=lambda c: (-float(c.score), float(c.km), int(c.auditor_id)))
    return candidates


def propose_team(task: Task, candidates):
    """
    稳定版组队逻辑：
    - need_expert=False：组长只需可带队
    - need_expert=True：仅组长必须 A 且可带队
    - 组员从剩余全部可用候选人中补齐，不强制 A
    - 硬指定人员：必须包含在团队中，但不限制系统继续自动补齐其他成员
    """
    if not candidates:
        return None

    need_n = max(1, int(getattr(task, "required_headcount", 1) or 1))
    need_members = max(0, need_n - 1)
    need_expert = bool(getattr(task, "need_expert", False))
    specified_names = set(_parse_names(getattr(task, "specified_auditors", None)))

    leader_pool = [c for c in candidates if bool(getattr(c, "can_lead_team", False))]
    if need_expert:
        leader_pool = [c for c in leader_pool if _norm(getattr(c, "group_level", "")) == "A"]

    if specified_names:
        # 如果硬指定中有人可带队，则优先从硬指定里选组长
        specified_leaders = [c for c in leader_pool if _norm(getattr(c, "auditor_name", "")) in specified_names]
        if specified_leaders:
            leader_pool = specified_leaders
        # 如果硬指定里没人可带队，则仍允许从普通 leader_pool 中选组长，但团队必须包含硬指定人员

    if not leader_pool:
        return None

    best_team = None
    best_score = None

    for leader in leader_pool:
        member_pool = [c for c in candidates if int(getattr(c, "auditor_id")) != int(getattr(leader, "auditor_id"))]

        must_have_members = [c for c in member_pool if _norm(getattr(c, "auditor_name", "")) in specified_names]
        # 若组长本身就是硬指定成员，则从 must_have 里自然不再重复出现
        remaining_pool = [c for c in member_pool if _norm(getattr(c, "auditor_name", "")) not in specified_names]

        # 团队必须包含全部硬指定人员（组长已占用的话自动满足其中一人）
        members = must_have_members[:]

        if len(members) > need_members:
            # 说明硬指定人数本身已经超过可容纳组员位
            continue

        still_need = need_members - len(members)
        members.extend(remaining_pool[:still_need])

        if len(members) < need_members:
            continue

        # 最终校验：所有硬指定成员都必须已在团队中
        team_names = {_norm(getattr(leader, "auditor_name", ""))}
        team_names.update({_norm(getattr(m, "auditor_name", "")) for m in members})
        if specified_names and not specified_names.issubset(team_names):
            continue

        avg_member_score = (sum(float(getattr(m, "score", 0.0)) for m in members) / len(members)) if members else 0.0
        team_score = round(float(getattr(leader, "score", 0.0)) + avg_member_score, 1)

        if specified_names:
            notes = "最终稳定逻辑：硬指定成员必须入组，但系统继续自动补齐其他成员"
        elif need_expert:
            notes = "最终稳定逻辑：仅组长要求A带队，组员按全部可用候选人补齐"
        else:
            notes = "标准逻辑：组长可带队，组员按全部可用候选人补齐"

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
