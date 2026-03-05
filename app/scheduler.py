from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Tuple, Optional
import re
import math

from sqlalchemy.orm import Session

from .config import (
    GAP_THRESHOLD_DAYS,
    CHAIN_GAP_DAYS,
    CHAIN_DISTANCE_KM,
    CHAIN_BONUS,
    EXPERT_MEMBERS_MUST_BE_A,
    TRAVEL_BUFFER_DAYS,
    DEFAULT_DISTANCE_KM,
    EXPERT_PREFERENCE_BONUS,
    DIST_PENALTY_DIVISOR,
    NEAR_BONUS_0_100,
    NEAR_BONUS_100_300,
    NEAR_BONUS_300_800,
    NEAR_BONUS_GT_800,
    LOAD_PENALTY_FACTOR,
    AUTO_DISTANCE_BY_GEO,
    FALLBACK_TO_DEFAULT_DISTANCE_IF_NO_GEO,
    OPT_ALPHA_DISTANCE,
    OPT_BETA_LOAD,
    OPT_GAMMA_WEEK,
    OPT_DELTA_CONT,
)
from .models import CityDistance, City, Schedule


def overlaps(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    return not (a_end < b_start or b_end < a_start)


def iso_week(d: date) -> Tuple[int, int]:
    y, w, _ = d.isocalendar()
    return (y, w)


def parse_names(s: Optional[str]) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[，,、\s\n]+", str(s).strip())
    return [p.strip() for p in parts if p.strip()]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_city_coord(db: Session, name: str):
    if not name:
        return None
    rec = db.query(City).filter(City.name == name).first()
    if not rec:
        return None
    return float(rec.lat), float(rec.lon)


def get_distance_km(db: Session, from_city: str, to_city: str) -> float:
    """距离优先：优先读 CityDistance；否则（若开启）用 City 坐标算直线距离并写回缓存；兜底 DEFAULT_DISTANCE_KM。"""
    if not from_city or not to_city:
        return float(DEFAULT_DISTANCE_KM)
    if from_city == to_city:
        return 0.0

    rec = db.query(CityDistance).filter(CityDistance.from_city == from_city, CityDistance.to_city == to_city).first()
    if rec:
        return float(rec.km)
    rec2 = db.query(CityDistance).filter(CityDistance.from_city == to_city, CityDistance.to_city == from_city).first()
    if rec2:
        return float(rec2.km)

    if AUTO_DISTANCE_BY_GEO:
        c1 = get_city_coord(db, from_city)
        c2 = get_city_coord(db, to_city)
        if c1 and c2:
            km = float(haversine_km(c1[0], c1[1], c2[0], c2[1]))
            try:
                db.add(CityDistance(from_city=from_city, to_city=to_city, km=km))
                db.commit()
            except Exception:
                db.rollback()
            return km

    return float(DEFAULT_DISTANCE_KM) if FALLBACK_TO_DEFAULT_DISTANCE_IF_NO_GEO else float(DEFAULT_DISTANCE_KM)


@dataclass
class Candidate:
    auditor_id: int
    auditor_name: str
    group_level: str
    can_lead_team: bool
    from_city: str
    km: float
    score: float
    explain: str


@dataclass
class TeamProposal:
    leader: Candidate
    members: List[Candidate]
    team_score: float
    notes: str


GROUP_POINTS = {"A": 6, "B": 6, "C": 2}


def task_has_any_schedule(db: Session, task_id: int) -> bool:
    """任务只要排过一次（任意人员），就视为已排班，不允许再次排。"""
    return db.query(Schedule.id).filter(Schedule.task_id == int(task_id)).first() is not None


def compute_from_city(a, task) -> str:
    """链式出发地：仅当 gap>=0 且 <=阈值时才使用 last_city，否则用 base_city。"""
    if a.last_task_end_city and a.last_task_end_date:
        gap = (task.start_date - a.last_task_end_date).days
        if 0 <= gap <= GAP_THRESHOLD_DAYS:
            return a.last_task_end_city
    return a.base_city


def hard_filter(a, task, schedules) -> Tuple[bool, str]:
    if a.status != "active":
        return False, "人员状态不可用"

    if task.required_gender and task.required_gender != "不限":
        if a.gender != task.required_gender:
            return False, "性别不匹配"

    specified = parse_names(task.specified_auditors)
    if specified and (a.name not in specified):
        return False, "不在指定人员名单"

    if task.need_expert and task.required_headcount <= 1:
        if a.group_level != "A":
            return False, "需要A组带队（单人任务）"

    # weekly cap
    y, w = iso_week(task.start_date)
    count_week = 0
    for s in schedules:
        if s.auditor_id != a.id:
            continue
        sy, sw = iso_week(s.start_date)
        if (sy, sw) == (y, w) and s.status in ("confirmed", "adjusted", "pre"):
            count_week += 1
    if count_week >= a.max_weekly_tasks:
        return False, "已达到每周频率上限"

    # 新增：last_end 与新任务 start 冲突拦截（即使 schedules 没记录也拦）
    if a.last_task_end_date:
        if (a.last_task_end_date + timedelta(days=TRAVEL_BUFFER_DAYS)) >= task.start_date:
            return False, f"与上次结束日期冲突(last_end={a.last_task_end_date})"

    # conflict with buffer (历史排班)
    task_start = task.start_date - timedelta(days=TRAVEL_BUFFER_DAYS)
    real_end = task.end_date or (task.start_date + timedelta(days=int(task.required_days or 1) - 1))
    task_end = real_end + timedelta(days=TRAVEL_BUFFER_DAYS)
    for s in schedules:
        if s.auditor_id != a.id:
            continue
        if overlaps(s.start_date, s.end_date, task_start, task_end) and s.status in ("confirmed", "adjusted", "pre"):
            return False, "时间冲突"

    return True, "OK"


def compute_score(a, task, from_city: str, km: float) -> Tuple[float, str]:
    group_base = GROUP_POINTS.get(a.group_level, 4)
    expert_bonus = 10 if (getattr(task, "need_expert", False) and a.group_level == "A") else 0

    preferred = parse_names(getattr(task, "preferred_experts", None))
    pref_bonus = EXPERT_PREFERENCE_BONUS if (preferred and a.name in preferred) else 0

    if km <= 100:
        near_bonus = NEAR_BONUS_0_100
    elif km <= 300:
        near_bonus = NEAR_BONUS_100_300
    elif km <= 800:
        near_bonus = NEAR_BONUS_300_800
    else:
        near_bonus = NEAR_BONUS_GT_800

    chain_bonus = 0
    if a.last_task_end_date and a.last_task_end_city:
        gap = (task.start_date - a.last_task_end_date).days
        if 0 <= gap <= CHAIN_GAP_DAYS and km <= CHAIN_DISTANCE_KM:
            chain_bonus = CHAIN_BONUS

    dist_penalty = km / float(DIST_PENALTY_DIVISOR)
    load_penalty = (a.monthly_cases * 1.0 + a.travel_days * 0.3) * float(LOAD_PENALTY_FACTOR)
    cont_penalty = max(0, a.continuous_days - 5) * 1.0

    score = 100.0 + group_base + expert_bonus + pref_bonus + near_bonus + chain_bonus - dist_penalty - load_penalty - cont_penalty
    explain = (
        f"出发地:{from_city}; 距离{km:.0f}km(扣{dist_penalty:.1f}), 就近+{near_bonus}, "
        f"组别+{group_base}, need_expert+{expert_bonus}, 指定专家(软)+{pref_bonus}, "
        f"链式+{chain_bonus}, 负荷-{load_penalty:.1f}, 连续-{cont_penalty:.1f}"
    )
    return round(score, 2), explain


def build_candidates(db: Session, task, auditors, schedules) -> List[Candidate]:
    # 任务已排班：直接不给候选，彻底避免重复排班/日历重复展示
    if task_has_any_schedule(db, task.id):
        return []

    res: List[Candidate] = []
    for a in auditors:
        ok, _ = hard_filter(a, task, schedules)
        if not ok:
            continue
        from_city = compute_from_city(a, task)
        km = get_distance_km(db, from_city, task.site_city)
        score, explain = compute_score(a, task, from_city, km)
        res.append(
            Candidate(
                auditor_id=a.id,
                auditor_name=a.name,
                group_level=a.group_level,
                can_lead_team=bool(a.can_lead_team),
                from_city=from_city,
                km=km,
                score=score,
                explain=explain,
            )
        )
    res.sort(key=lambda x: x.score, reverse=True)
    return res


def propose_team(task, candidates: List[Candidate]) -> Optional[TeamProposal]:
    n = max(1, int(task.required_headcount or 1))

    if n == 1:
        if not candidates:
            return None
        return TeamProposal(leader=candidates[0], members=[], team_score=candidates[0].score, notes="单人任务")

    leader_pool = [c for c in candidates if c.can_lead_team]
    if task.need_expert:
        leader_pool = [c for c in leader_pool if c.group_level == "A"]
    if not leader_pool:
        return None
    leader = leader_pool[0]

    member_pool = [c for c in candidates if c.auditor_id != leader.auditor_id]
    if task.need_expert and EXPERT_MEMBERS_MUST_BE_A:
        member_pool = [c for c in member_pool if c.group_level == "A"]
    members = member_pool[: max(0, n - 1)]
    if len(members) < n - 1:
        return None

    team_score = leader.score + sum(m.score for m in members) / max(1, len(members))
    notes = "组队：负责人可带队" + ("；负责人A组" if task.need_expert else "")
    if task.need_expert and EXPERT_MEMBERS_MUST_BE_A:
        notes += "；组员也要求A组"
    return TeamProposal(leader=leader, members=members, team_score=round(team_score, 2), notes=notes)


def team_objective(team: TeamProposal, auditor_lookup: dict, avg_monthly_cases: float, batch_week_counts: dict) -> float:
    kms = [team.leader.km] + [m.km for m in team.members]
    avg_km = sum(kms) / max(1, len(kms))

    load_pen = 0.0
    for c in [team.leader] + team.members:
        a = auditor_lookup.get(c.auditor_id)
        if not a:
            continue
        load_pen += max(0.0, float(a.monthly_cases) - avg_monthly_cases)

    week_pen = 0.0
    for c in [team.leader] + team.members:
        week_pen += float(batch_week_counts.get(c.auditor_id, 0))

    cont_pen = 0.0
    for c in [team.leader] + team.members:
        a = auditor_lookup.get(c.auditor_id)
        if not a:
            continue
        cont_pen += max(0.0, float(a.continuous_days) - 5.0)

    obj = (-team.team_score) + OPT_ALPHA_DISTANCE * (avg_km / 100.0) + OPT_BETA_LOAD * load_pen + OPT_GAMMA_WEEK * week_pen + OPT_DELTA_CONT * cont_pen
    return float(obj)
