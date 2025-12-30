from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, Text
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    pass

class Shot(Base):
    __tablename__ = "shots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|processing|done|failed
    video_path: Mapped[str] = mapped_column(Text)                  # caminho no container
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
