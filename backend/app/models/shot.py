import enum
from sqlalchemy import String, DateTime, Enum, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db import Base


class ShotStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


class Shot(Base):
    __tablename__ = "shots"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    video_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[ShotStatus] = mapped_column(Enum(ShotStatus), default=ShotStatus.queued, index=True)

    created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped["DateTime"] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
