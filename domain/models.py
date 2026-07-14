import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(8), nullable=False, index=True)
    score = Column(Float, nullable=False)
    faithful = Column(Boolean, nullable=False)
    reasoning = Column(Text, nullable=False)
    model = Column(String(100), nullable=False)
    latency_seconds = Column(Float, nullable=False)
    unsupported_claims = Column(JSONB, nullable=False, default=list)
    omitted_information = Column(JSONB, nullable=False, default=list)
    contradictions = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
