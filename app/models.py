from sqlalchemy import Boolean, Column, Date, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from .db import Base, initialize_database


class Auditor(Base):
    __tablename__ = "auditors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    gender = Column(String, nullable=False, default="男")
    group_level = Column(String, nullable=False, default="B")
    can_lead_team = Column(Boolean, nullable=False, default=False)
    base_city = Column(String, nullable=False)
    max_weekly_tasks = Column(Integer, nullable=False, default=1)

    monthly_cases = Column(Integer, nullable=False, default=0)
    travel_days = Column(Integer, nullable=False, default=0)
    continuous_days = Column(Integer, nullable=False, default=0)

    last_task_end_city = Column(String, nullable=True)
    last_task_end_date = Column(Date, nullable=True)

    status = Column(String, nullable=False, default="active")

    schedules = relationship("Schedule", back_populates="auditor")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    project_name = Column(String, nullable=False)
    customer_name = Column(String, nullable=True)

    need_expert = Column(Boolean, nullable=False, default=False)
    required_headcount = Column(Integer, nullable=False, default=1)
    required_days = Column(Integer, nullable=False, default=1)

    specified_auditors = Column(Text, nullable=True)
    preferred_experts = Column(Text, nullable=True)
    required_gender = Column(String, nullable=False, default="不限")

    site_city = Column(String, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=True)

    schedules = relationship("Schedule", back_populates="task")


class Schedule(Base):
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    auditor_id = Column(Integer, ForeignKey("auditors.id"), nullable=False)

    role = Column(String, nullable=False, default="member")
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)

    travel_from_city = Column(String, nullable=False)
    travel_to_city = Column(String, nullable=False)
    distance_km = Column(Float, nullable=False, default=0.0)

    score = Column(Float, nullable=False, default=0.0)
    status = Column(String, nullable=False, default="confirmed")

    task = relationship("Task", back_populates="schedules")
    auditor = relationship("Auditor", back_populates="schedules")


class CityDistance(Base):
    __tablename__ = "city_distances"
    __table_args__ = (UniqueConstraint("from_city", "to_city", name="uix_from_to"),)

    id = Column(Integer, primary_key=True, index=True)
    from_city = Column(String, nullable=False)
    to_city = Column(String, nullable=False)
    km = Column(Float, nullable=False, default=0.0)


class City(Base):
    __tablename__ = "cities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)


# All model metadata is registered at this point, so startup can safely create
# the schema before the login page tries to read the authentication table.
initialize_database()
