from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from domain.models import Base
from domain.settings import settings

_engine = None
_session_factory = None


def create_session_factory() -> async_sessionmaker[AsyncSession]:
    global _engine, _session_factory
    if _session_factory is not None:
        return _session_factory
    _engine = create_async_engine(settings.database_url, echo=False)
    _session_factory = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )
    return _session_factory


async def init_db():
    create_session_factory()
    assert _engine is not None
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
