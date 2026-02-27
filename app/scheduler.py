from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Tuple, Optional
import re
from sqlalchemy.orm import Session
from .config import GAP_THRESHOLD_DAYS, CHAIN_GAP_DAYS, CHAIN_DISTANCE_KM, CHAIN_BONUS, EXPERT_MEMBERS_MUST_BE_A, TRAVEL_BUFFER_DAYS, DEFAULT_DISTANCE_KM, EXPERT_PREFERENCE_BONUS, DIST_PENALTY_DIVISOR, NEAR_BONUS_0_100, NEAR_BONUS_100_300, NEAR_BONUS_300_800, NEAR_BONUS_GT_800, LOAD_PENALTY_FACTOR, AUTO_DISTANCE_BY_GEO, FALLBACK_TO_DEFAULT_DISTANCE_IF_NO_GEO
from .models import CityDistance

def overlaps(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    return not (a_end < b_start or b_end < a_start)

def iso_week(d: date) -> Tuple[int, int]:
    y, w, _ = d.isocalendar()
    return (y, w)

def parse_names(s: Optional[str]) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[，,、\s\n]+", s.strip())
    return [p.strip() for p in parts if p.strip()]

def get_distance_km(db: Session, from_city: str, to_city: str) -> float:
    if not from_city or not to_city:
        return float(DEFAULT_DISTANCE_KM)
    if from_city == to_city:
        return 0.0

    # 1) 先查缓存的城市距离表（人工维护/历史计算）
    rec = db.query(CityDistance).filter(CityDistance.from_city == from_city, CityDistance.to_city == to_city).first()
    if rec:
        return float(rec.km)
    rec2 = db.query(CityDistance).filter(CityDistance.from_city == to_city, CityDistance.to_city == from_city).first()
    if rec2:
        return float(rec2.km)

    # 2) 自动：若城市坐标表中存在两端城市，则计算直线距离，并写回 CityDistance 做缓存
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

    # 3) 兜底：默认距离（避免系统无结果）
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

def compute_from_city(a, task) -> str:
    if a.last_task_end_city and a.last_task_end_date:
        gap = (task.start_date - a.last_task_end_date).days
        if gap <= GAP_THRESHOLD_DAYS:
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
        if (sy, sw) == (y, w) and s.status in ("confirmed","adjusted","pre"):
            count_week += 1
    if count_week >= a.max_weekly_tasks:
        return False, "已达到每周频率上限"
    # conflict with buffer
    task_start = task.start_date - timedelta(days=TRAVEL_BUFFER_DAYS)
    task_end = task.start_date + timedelta(days=task.required_days - 1 + TRAVEL_BUFFER_DAYS)
    for s in schedules:
        if s.auditor_id != a.id:
            continue
        if overlaps(s.start_date, s.end_date, task_start, task_end) and s.status in ("confirmed","adjusted","pre"):
            return False, "时间冲突"
    return True, "OK"

def compute_score(a, task, from_city: str, km: float) -> Tuple[float, str]:
    # 目标：在无特殊要求时，“距离越近分越高”是第一优先级
    group_base = GROUP_POINTS.get(a.group_level, 4)

    # need_expert（需要A带队）仍保留：A组执行更匹配
    expert_bonus = 10 if (getattr(task, "need_expert", False) and a.group_level == "A") else 0

    # 软指定专家/老师：命中 preferred_experts 则加分（不强制）
    preferred = parse_names(getattr(task, "preferred_experts", None))
    pref_bonus = EXPERT_PREFERENCE_BONUS if (preferred and a.name in preferred) else 0

    # 就近加成（更大权重）
    if km <= 100:
        near_bonus = NEAR_BONUS_0_100
    elif km <= 300:
        near_bonus = NEAR_BONUS_100_300
    elif km <= 800:
        near_bonus = NEAR_BONUS_300_800
    else:
        near_bonus = NEAR_BONUS_GT_800

    # 链式加成
    chain_bonus = 0
    if a.last_task_end_date and a.last_task_end_city:
        gap = (task.start_date - a.last_task_end_date).days
        if gap <= CHAIN_GAP_DAYS and km <= CHAIN_DISTANCE_KM:
            chain_bonus = CHAIN_BONUS

    # 距离惩罚（更敏感）
    dist_penalty = km / float(DIST_PENALTY_DIVISOR)

    # 负荷惩罚（次要）
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
    res: List[Candidate] = []
    for a in auditors:
        ok, _ = hard_filter(a, task, schedules)
        if not ok:
            continue
        from_city = compute_from_city(a, task)
        km = get_distance_km(db, from_city, task.site_city)
        score, explain = compute_score(a, task, from_city, km)
        res.append(Candidate(
            auditor_id=a.id,
            auditor_name=a.name,
            group_level=a.group_level,
            can_lead_team=bool(a.can_lead_team),
            from_city=from_city,
            km=km,
            score=score,
            explain=explain
        ))
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
    members = member_pool[: max(0, n-1)]
    if len(members) < n-1:
        return None

    team_score = leader.score + sum(m.score for m in members) / max(1, len(members))
    notes = "组队：负责人可带队" + ("；负责人A组" if task.need_expert else "")
    if task.need_expert and EXPERT_MEMBERS_MUST_BE_A:
        notes += "；组员也要求A组"
    return TeamProposal(leader=leader, members=members, team_score=round(team_score,2), notes=notes)

from .config import OPT_ALPHA_DISTANCE, OPT_BETA_LOAD, OPT_GAMMA_WEEK, OPT_DELTA_CONT

def team_objective(team: TeamProposal, auditor_lookup: dict, avg_monthly_cases: float, batch_week_counts: dict) -> float:
    """越小越好：综合 总距离+负荷集中+本周集中+连续出差 ；同时用 team_score 作为质量/匹配收益（取负号）"""
    # 距离：负责人+组员平均距离
    kms = [team.leader.km] + [m.km for m in team.members]
    avg_km = sum(kms) / max(1, len(kms))

    # 负荷：月度院次超过均值的惩罚（负责人+组员）
    load_pen = 0.0
    for c in [team.leader] + team.members:
        a = auditor_lookup.get(c.auditor_id)
        if not a:
            continue
        load_pen += max(0.0, float(a.monthly_cases) - avg_monthly_cases)

    # 本周集中惩罚：在一次批量排班中同一周被用的次数（临时计数）
    week_pen = 0.0
    for c in [team.leader] + team.members:
        week_pen += float(batch_week_counts.get(c.auditor_id, 0))

    # 连续出差惩罚（使用auditor当前连续天数）
    cont_pen = 0.0
    for c in [team.leader] + team.members:
        a = auditor_lookup.get(c.auditor_id)
        if not a:
            continue
        cont_pen += max(0.0, float(a.continuous_days) - 5.0)

    # 目标函数：小为优
    obj = (-team.team_score) + OPT_ALPHA_DISTANCE * (avg_km / 100.0) + OPT_BETA_LOAD * load_pen + OPT_GAMMA_WEEK * week_pen + OPT_DELTA_CONT * cont_pen
    return float(obj)

import math

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # 地球半径（km）
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_city_coord(db: Session, name: str):
    if not name:
        return None
    rec = db.query(City).filter(City.name == name).first()
    if not rec:
        return None
    return float(rec.lat), float(rec.lon)
