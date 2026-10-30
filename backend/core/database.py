"""
Database Configuration & ORM Models
Multimodal Attendance & Engagement Agent

Sets up SQLite database engine, session factory, and declarative models
for sessions, attendance logs, and student performance tracking.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./attendance_engagement.db")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Session(Base):
    """Represents a classroom teaching session."""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_uid = Column(String(64), unique=True, index=True)
    teacher_id = Column(String(64), index=True)
    class_name = Column(String(128))
    date = Column(DateTime, default=datetime.utcnow)
    average_engagement = Column(Float, default=0.0)
    attendance_rate = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)

class StudentRecord(Base):
    """Represents historical student performance and attendance per session."""
    __tablename__ = "student_records"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(64), index=True)
    student_name = Column(String(128))
    session_id = Column(String(64), index=True)
    presence_duration_seconds = Column(Float, default=0.0)
    attendance_status = Column(String(32), default="ABSENT")
    attention_score = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)

class SessionPerson(Base):
    """Represents an individual student participant associated with a session."""
    __tablename__ = "session_persons"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    person_id = Column(String(64), index=True)
    name = Column(String(128), nullable=True)
    presence_duration = Column(Float, default=0.0)
    attendance_status = Column(String(32), default="ABSENT")
    average_attention = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class PersonMetric(Base):
    """Represents fine-grained temporal engagement and attention telemetry per participant."""
    __tablename__ = "person_metrics"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    person_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    attention = Column(Float, default=0.0)
    emotion = Column(String(32), default="neutral")

# Create all tables on startup
Base.metadata.create_all(bind=engine)

