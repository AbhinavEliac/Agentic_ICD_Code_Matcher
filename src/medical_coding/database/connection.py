"""Database engine initialization, connection pooling, and session lifecycle management."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from medical_coding.config.settings import Settings, get_settings
from medical_coding.database.models import Base
from medical_coding.utils.logging import get_logger

logger = get_logger(__name__)

_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def get_engine(settings: Settings | None = None) -> Engine:
    """Initialize or return the cached singleton SQLAlchemy Engine."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    cfg = settings or get_settings()
    db_url = cfg.database_url

    # Ensure parent directory exists for SQLite database files
    if db_url.startswith("sqlite:///"):
        sqlite_path_str = db_url.replace("sqlite:///", "")
        if sqlite_path_str != ":memory:":
            db_file_path = Path(sqlite_path_str)
            db_file_path.parent.mkdir(parents=True, exist_ok=True)

    connect_args: dict[str, Any] = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    # Enable SQLite WAL mode and foreign key constraints on connection
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys = ON;")
                cursor.execute("PRAGMA journal_mode = WAL;")
                cursor.execute("PRAGMA synchronous = NORMAL;")
            finally:
                cursor.close()

    _ENGINE = engine
    return _ENGINE


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    """Initialize or return the cached sessionmaker."""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is not None:
        return _SESSION_FACTORY

    engine = get_engine(settings=settings)
    _SESSION_FACTORY = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    return _SESSION_FACTORY


def get_scoped_session(settings: Settings | None = None) -> scoped_session[Session]:
    """Return a thread-local scoped session."""
    factory = get_session_factory(settings=settings)
    return scoped_session(factory)


@contextmanager
def get_db_session(settings: Settings | None = None) -> Generator[Session]:
    """Context manager for database transactional sessions with automatic rollback on error."""
    factory = get_session_factory(settings=settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(settings: Settings | None = None) -> None:
    """Create all configured database tables if they do not already exist and migrate columns."""
    from sqlalchemy import inspect, text

    engine = get_engine(settings=settings)
    Base.metadata.create_all(bind=engine)

    # Safely migrate new columns to diagnosis_records if it was created under previous schema
    try:
        inspector = inspect(engine)
        if "diagnosis_records" in inspector.get_table_names():
            columns = {col["name"] for col in inspector.get_columns("diagnosis_records")}
            with engine.connect() as conn:
                if "icd10cm" not in columns:
                    conn.execute(text("ALTER TABLE diagnosis_records ADD COLUMN icd10cm VARCHAR(32);"))
                if "icdo" not in columns:
                    conn.execute(text("ALTER TABLE diagnosis_records ADD COLUMN icdo VARCHAR(32);"))
                if "cpt" not in columns:
                    conn.execute(text("ALTER TABLE diagnosis_records ADD COLUMN cpt VARCHAR(32);"))
                conn.commit()
    except Exception as exc:
        logger.warning("Database column migration check notice: %s", exc)

    logger.info("Initialized database schema successfully at %s", engine.url)

